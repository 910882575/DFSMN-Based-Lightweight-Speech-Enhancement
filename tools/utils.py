#!/usr/bin/env python
# coding=utf-8

import soundfile as sf


def audioread(path,):
    return sf.read(path)

def audiowrite(path, data, fs):
    return sf.write(path, data, fs)
