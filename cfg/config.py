# config.py
import yaml
import platform

def load_cfg_from_file(file_path):
    """YAML 파일에서 설정을 불러옵니다."""
    with open(file_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    return yaml_cfg

def load_cfg_and_update_args(args):
    """YAML 설정을 args에 병합합니다."""
    yaml_cfg = load_cfg_from_file(args.cfg)
    
    os_type = platform.system().lower()
    
    assert os_type in ['windows', 'linux'], f'Unsupported OS: {os_type}'
    
    if os_type == 'windows':
        args.dist_backend = 'gloo'
    elif os_type == 'linux':
        args.dist_backend = 'nccl'
    
    print(f'Os type: {os_type}')
    print(f'Using {args.dist_backend} backend for distributed training')
    
    for key, value in yaml_cfg.items():
        # argparse 인자 중에 해당 키가 있으면 값을 업데이트
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            # argparse에 없는 인자는 새로 추가
            setattr(args, key, value)
    return args
