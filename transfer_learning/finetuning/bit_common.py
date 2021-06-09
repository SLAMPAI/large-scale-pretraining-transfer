# Copyright 2020 Google LLC
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

# Lint as: python3
# coding: utf-8

import argparse
import logging
import logging.config
import os

from . import bit_hyperrule


def argparser(known_models):
  parser = argparse.ArgumentParser(description="Fine-tune BiT-M model.")
  parser.add_argument("--logdir", required=True,
                      help="Where to log training info (small).")
  parser.add_argument("--examples_per_class", type=int, default=None,
                      help="For the few-shot variant, use this many examples "
                      "per class only.")
  parser.add_argument("--examples_per_class_seed", type=int, default=0,
                      help="Random seed for selecting examples.")
  return parser


def setup_logger(args):
  """Creates and returns a fancy logger."""
  # return logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
  # Why is setting up proper logging so !@?#! ugly?
  os.makedirs(os.path.join(args.logdir), exist_ok=True)
  logging.config.dictConfig({
      "version": 1,
      "disable_existing_loggers": False,
      "formatters": {
          "standard": {
              "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
          },
      },
      "handlers": {
          "stderr": {
              "level": "INFO",
              "formatter": "standard",
              "class": "logging.StreamHandler",
              "stream": "ext://sys.stderr",
          },
          "logfile": {
              "level": "DEBUG",
              "formatter": "standard",
              "class": "logging.FileHandler",
              "filename": os.path.join(args.logdir, "train.log"),
              "mode": "a",
          }
      },
      "loggers": {
          "": {
              "handlers": ["stderr", "logfile"],
              "level": "DEBUG",
              "propagate": True
          },
      }
  })
  logger = logging.getLogger(__name__)
  logger.flush = lambda: [h.flush() for h in logger.handlers]
  logger.info(args)
  return logger
