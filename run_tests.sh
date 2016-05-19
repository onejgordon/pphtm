#!/bin/bash
module=${1:-all}

sudo ./runtests.py testing/ $module