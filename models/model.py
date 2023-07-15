import argparse
import logging
from contextlib import suppress
from functools import partial
import torch

from models.timm.data import resolve_data_config
from models.timm.layers import apply_test_time_pool
from models.timm.models import create_model
from models.timm.utils import set_jit_fuser, ParseKwargs

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('inference')


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
    parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                        help='path to dataset (*deprecated*, use --data-dir)')
    parser.add_argument('--data-dir', metavar='DIR',
                        help='path to dataset (root dir)')
    parser.add_argument('--dataset', metavar='NAME', default='',
                        help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
    parser.add_argument('--split', metavar='NAME', default='validation',
                        help='dataset split (default: validation)')
    parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                        help='model architecture (default: resnet50)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--img-size', default=None, type=int,
                        metavar='N', help='Input image dimension, uses model default if empty')
    parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                        help='Image input channels (default: None => 3)')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--use-train-size', action='store_true', default=False,
                        help='force use of train input size, even when test size is specified in pretrained cfg')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop pct')
    parser.add_argument('--crop-mode', default=None, type=str,
                        metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number classes in dataset')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')
    parser.add_argument('--log-freq', default=10, type=int,
                        metavar='N', help='batch logging frequency (default: 10)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--num-gpu', type=int, default=1,
                        help='Number of GPUS to use')
    parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                        help='enable test time pool')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--device', default='cuda', type=str,
                        help="Device (accelerator) to use.")
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use Native AMP for mixed precision training')
    parser.add_argument('--amp-dtype', default='float16', type=str,
                        help='lower precision AMP dtype (default: float16)')
    parser.add_argument('--fuser', default='', type=str,
                        help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)

    parser.add_argument('--results-dir',type=str, default=None,
                        help='folder for output results')
    parser.add_argument('--results-file', type=str, default=None,
                        help='results filename (relative to results-dir)')
    parser.add_argument('--results-format', type=str, nargs='+', default=['csv'],
                        help='results format (one of "csv", "json", "json-split", "parquet")')
    parser.add_argument('--results-separate-col', action='store_true', default=False,
                        help='separate output columns per result index.')
    parser.add_argument('--topk', default=1, type=int,
                        metavar='N', help='Top-k to output to CSV')
    parser.add_argument('--fullname', action='store_true', default=False,
                        help='use full sample name in output (not just basename).')
    parser.add_argument('--filename-col', default='filename',
                        help='name for filename / sample name column')
    parser.add_argument('--index-col', default='index',
                        help='name for output indices column(s)')
    parser.add_argument('--output-col', default=None,
                        help='name for logit/probs output column(s)')
    parser.add_argument('--output-type', default='prob',
                        help='output type colum ("prob" for probabilities, "logit" for raw logits)')
    parser.add_argument('--exclude-output', action='store_true', default=False,
                        help='exclude logits/probs from results, just indices. topk must be set !=0.')
    
    return parser.parse_args(args = [
        '--model', 'None',
        '--pretrained', 'True',
    ])

def define_model(model_name, depth) -> object:
    
    if "eva" in model_name:
        args = get_args()
        if model_name == "eva02-clip-enormous":
            args.model = "eva02_enormous_patch14_clip_224.laion2b_plus"
            
        elif model_name == "eva02-clip-large":
            args.model = "eva02_large_patch14_clip_224.merged2b"
        
        elif model_name == "eva02-clip-base":
            args.model = "eva02_base_patch16_clip_224.merged2b"
        
        elif model_name == "eva-clip-giant":
            args.model = "eva_giant_patch14_clip_224.merged2b"
            
        else:
            raise AssertionError(f"Image Bind is in preparation.")
        
        args.pretrained = args.pretrained or not args.checkpoint
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # resolve AMP arguments based on PyTorch / Apex availability
        amp_autocast = suppress
        if args.amp:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            assert args.amp_dtype in ('float16', 'bfloat16')
            amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
            _logger.info('Running inference in mixed precision with native PyTorch AMP.')
        else:
            _logger.info('Running inference in float32. AMP not enabled.')

        if args.fuser:
            set_jit_fuser(args.fuser)

        # create model
        in_chans = 3
        if args.in_chans is not None:
            in_chans = args.in_chans
        elif args.input_size is not None:
            in_chans = args.input_size[0]
        model = create_model(
                        args.model,
                        num_classes=args.num_classes,
                        in_chans=in_chans,
                        pretrained=args.pretrained,
                        checkpoint_path=args.checkpoint,
                        extracting_depth=depth,
                        **args.model_kwargs,
                    )
        if args.num_classes is None:
            assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            args.num_classes = model.num_classes

        _logger.info(
            f'Model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}')

        data_config = resolve_data_config(vars(args), model=model)
        test_time_pool = False
        if args.test_pool:
            model, test_time_pool = apply_test_time_pool(model, data_config)
            
        return model
    
    elif model_name == 'clip-convnext':
        args = get_args()
        args.model = 'convnext_xxlarge.clip_laion2b_soup_ft_in1k'
        args.pretrained = args.pretrained or not args.checkpoint
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        # resolve AMP arguments based on PyTorch / Apex availability
        amp_autocast = suppress
        if args.amp:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            assert args.amp_dtype in ('float16', 'bfloat16')
            amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
            _logger.info('Running inference in mixed precision with native PyTorch AMP.')
        else:
            _logger.info('Running inference in float32. AMP not enabled.')

        if args.fuser:
            set_jit_fuser(args.fuser)

        # create model
        in_chans = 3
        if args.in_chans is not None:
            in_chans = args.in_chans
        elif args.input_size is not None:
            in_chans = args.input_size[0]
        model = create_model(
                        args.model,
                        num_classes=args.num_classes,
                        in_chans=in_chans,
                        pretrained=args.pretrained,
                        checkpoint_path=args.checkpoint,
                        extracting_depth=depth,
                        **args.model_kwargs,
                    )
        if args.num_classes is None:
            assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
            args.num_classes = model.num_classes

        _logger.info(
            f'Model {args.model} created, param count: {sum([m.numel() for m in model.parameters()])}')

        data_config = resolve_data_config(vars(args), model=model)
        test_time_pool = False
        if args.test_pool:
            model, test_time_pool = apply_test_time_pool(model, data_config)
        
        return model
    
    elif model_name=='InternImage':
            sys.path.append("../")
            from classification.models import build_model
            import yaml
            from attrdict import AttrDict
            
            with open('/mount/nfs6/takuyamatsuyama/HugeDNNs-Encoding/InternImage/classification/configs/internimage_g_22kto1k_384.yaml') as f:
                config = yaml.safe_load(f.read())
            config = AttrDict(config)
            print(config)
            model = build_model(config)
            checkpoint = torch.load("/mount/nfs6/takuyamatsuyama/data/checkpoint/internimage_g_pretrainto22k_384.pth", map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            model.cuda()
            print(model)
            
    elif model_name=='ONE-PEACE':
        sys.path.append('/mount/nfs6/takuyamatsuyama/HugeDNNs-Encoding/ONE-PEACE')
        from one_peace.models import from_pretrained

        model = from_pretrained("ONE-PEACE", device='cuda:0', dtype="float32")

    elif model_name=='imagebind':
        raise AssertionError(f"Image Bind is in preparation.")
    else:
        raise AssertionError(f"Model name '{model_name}' does not exit")