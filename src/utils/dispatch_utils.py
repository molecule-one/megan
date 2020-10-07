""" Utils to provide additional functionality to dispatch exe commands together with the gin configuration library """

# requirements: gin
import argparse
import logging
import os
import sys
from contextlib import contextmanager
from typing import Callable, Dict, List, TextIO, Optional

import gin

from src import CONFIGS_DIR, configure_logger, LOG_LEVEL

logger = logging.getLogger(__name__)


def parse_config_file(path: str):
    gin.parse_config_file(path, skip_unknown=True)
    logger.info('Parsed gin config from {}'.format(path))


def save_current_config(output_path: str):
    with open(output_path, 'w') as f:
        # noinspection PyProtectedMember
        for k, args in gin.config._CONFIG.items():
            full_function_path = k[1]
            function_name = full_function_path.split('.')[-1]

            for arg_name, arg_val in args.items():
                if isinstance(arg_val, str):
                    arg_val = '"{}"'.format(arg_val)
                f.write('{}.{} = {}\n'.format(function_name, arg_name, arg_val))
            f.write('\n')


def parse_config_key(arg: str):
    if arg.endswith('.gin'):
        return arg  # assume this is a concrete path
    else:
        return os.path.join(CONFIGS_DIR, arg + ".gin")  # assume this is a name of a file in configs dir


def _argparse_gin_bindings(parser) -> Dict:
    keys = {}
    # noinspection PyProtectedMember
    for k, args in gin.config._CONFIG.items():
        for arg_name, arg_val in args.items():
            if isinstance(arg_val, bool):
                if arg_val is True:
                    parser.add_argument('--no_' + arg_name, default=arg_val,
                                        help="default: {}".format(not arg_val), action='store_false')
                else:
                    parser.add_argument('--' + arg_name, default=arg_val,
                                        help="default: {}".format(arg_val), action='store_true')
            else:
                parser.add_argument('--' + arg_name, default=arg_val, type=type(arg_val),
                                    help="default: {}".format(arg_val))
            binding_key = '.'.join([k[1].split('.')[-1], arg_name])
            keys[arg_name] = binding_key
    parser.set_defaults(feature=True)
    return keys


def log_current_config():
    """
    Logs information about currently loaded gin config to the logger
    """
    # noinspection PyProtectedMember
    logs = []
    for k, args in gin.config._CONFIG.items():
        for arg_name, arg_val in args.items():
            binding_key = '.'.join([k[1].split('.')[-1], arg_name])
            logs.append('{} = {}'.format(binding_key, arg_val))
    config_log = 'gin-bound config values:\n' + '\n'.join(logs)
    logger.info(config_log)


class Fork(object):
    def __init__(self, main_output: TextIO, fork_outputs: List[TextIO]):
        self.main_output = main_output
        self.fork_outputs = fork_outputs

    def write(self, data):
        self.main_output.write(data)

        for output in self.fork_outputs:
            output.write(data)

    def flush(self):
        self.main_output.flush()
        for output in self.fork_outputs:
            output.flush()


@contextmanager
def replace_standard_stream(stream_name, file_):
    stream = getattr(sys, stream_name)
    setattr(sys, stream_name, file_)
    try:
        yield
    finally:
        setattr(sys, stream_name, stream)


def run_with_redirection(func: Callable, stdout_path: Optional[str] = None, stderr_path: Optional[str] = None):
    if stdout_path is None:
        stdout_path = os.path.join(os.environ['LOGS_DIR'], 'stdout.txt')
    if stderr_path is None:
        stderr_path = os.path.join(os.environ['LOGS_DIR'], 'stderr.txt')

    def func_wrapper(*args, **kwargs):
        with open(stdout_path, 'a', 1) as out_dst:
            with open(stderr_path, 'a', 1) as err_dst:
                out_fork = Fork(sys.stdout, [out_dst])
                err_fork = Fork(sys.stderr, [err_dst])
                with replace_standard_stream('stderr', err_fork):
                    with replace_standard_stream('stdout', out_fork):
                        func(*args, **kwargs)

    return func_wrapper


def dispatch_configurable_command(function: Callable[[str], None]):
    """
    Dispatches a command with optional arguments same as bindings in the provided configs.
    Assumes that first 2 positional arguments are config path and model save path
    :param function: callable to use (with a single positional argument: save_path)
    """
    # parse arguments and/or print help
    parser = argparse.ArgumentParser(description=function.__doc__)

    conf_arg_key = 'conf_path'
    save_arg_key = 'save_path'
    bind_arg_key = 'additional_bindings'

    parser.add_argument(conf_arg_key, type=str, help="key or path to gin config file")
    parser.add_argument(save_arg_key, type=str, help="directory to save model")
    parser.add_argument(bind_arg_key, help="additional bindings", nargs='*')

    argv1 = sys.argv[1].lower()
    if argv1 == '-h' or argv1 == '--help':
        parser.parse_args()
        return

    conf_path = sys.argv[1]

    conf_path = parse_config_key(conf_path)
    parse_config_file(conf_path)

    binding_keys = _argparse_gin_bindings(parser)

    parser.parse_args()

    argv2 = sys.argv[2].lower()
    if argv2 == '-h' or argv2 == '--help':
        parser.parse_args()
        return

    save_path = sys.argv[2]

    namespace = parser.parse_args()
    for arg_key, arg_val in namespace.__dict__.items():
        if arg_key.startswith('no_'):
            arg_key = arg_key[3:]

        if arg_key in binding_keys:
            gin.bind_parameter(binding_keys[arg_key], arg_val)
        elif arg_key == bind_arg_key:
            for val in arg_val:
                s = val.split('=')
                k, v = s[0], eval(s[1])
                gin.bind_parameter(k, v)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    configure_logger(name='', console_logging_level=LOG_LEVEL,
                     logs_dir=os.path.join(save_path, 'logs'),
                     reset=True)

    if int(os.environ.get('DEPLOY', 0)):
        function(save_path)
    else:
        run_with_redirection(function,
                             os.path.join(save_path, "stdout.txt"),
                             os.path.join(save_path, "stderr.txt"))(save_path)
