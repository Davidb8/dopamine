# coding=utf-8
# Copyright 2023 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compatibility layer for tensorflow functionality.

This module provides replacements for TensorFlow functionality used in the
JAX implementation, making it compatible with Python 3.13 which doesn't support
TensorFlow.
"""

import os
import pathlib
import time
from typing import Dict, Optional, TextIO, Union


class FileWriter:
    """A simple file writer to replace tf.summary.create_file_writer."""
    
    def __init__(self, logdir: str):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self.scalar_file = open(os.path.join(logdir, 'scalars.csv'), 'a')
        self.scalar_file.write('timestamp,step,tag,value\n')
        self.scalar_file.flush()
    
    def __del__(self):
        if hasattr(self, 'scalar_file'):
            self.scalar_file.close()
    
    def as_default(self):
        """Context manager support (no-op for compatibility)."""
        class DummyContext:
            def __enter__(self):
                return None
            
            def __exit__(self, *args):
                return None
        
        return DummyContext()


class SummaryNamespace:
    """A namespace for summary operations."""
    
    @staticmethod
    def scalar(tag: str, value: float, step: int):
        """Log a scalar value."""
        logdir = get_active_logdir()
        if logdir:
            scalar_file = os.path.join(logdir, 'scalars.csv')
            with open(scalar_file, 'a') as f:
                f.write(f"{time.time()},{step},{tag},{value}\n")


class IONamespace:
    """A namespace for I/O operations."""
    
    class gfile:
        """A namespace for file operations."""
        
        @staticmethod
        def exists(path: str) -> bool:
            """Check if a path exists."""
            return os.path.exists(path)


class ErrorsNamespace:
    """A namespace for error handling."""
    
    class NotFoundError(FileNotFoundError):
        """Error for files not found."""
        pass


class CompatNamespace:
    """A namespace for compatibility operations."""
    
    class v1:
        """A namespace for v1 compatibility."""
        
        @staticmethod
        def enable_v2_behavior():
            """No-op for compatibility."""
            pass


# Global state to keep track of the active logdir
_active_logdir = None

def get_active_logdir() -> Optional[str]:
    """Get the active logdir."""
    return _active_logdir

def create_file_writer(logdir: str) -> FileWriter:
    """Create a file writer."""
    global _active_logdir
    _active_logdir = logdir
    return FileWriter(logdir)


# TensorFlow compatibility namespace
class TensorFlowCompat:
    """A namespace for TensorFlow compatibility."""
    
    summary = SummaryNamespace()
    io = IONamespace()
    errors = ErrorsNamespace()
    compat = CompatNamespace()


# Export the compatibility layer as tf
tf = TensorFlowCompat() 