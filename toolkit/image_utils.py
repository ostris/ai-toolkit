# ref https://github.com/scardine/image_size/blob/master/get_image_size.py
import atexit
import collections
import json
import os
import io
import struct
import threading
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from diffusers import AutoencoderTiny
from PIL import Image as PILImage

FILE_UNKNOWN = "Sorry, don't know how to get size for this file."


class UnknownImageFormat(Exception):
    pass


types = collections.OrderedDict()
BMP = types['BMP'] = 'BMP'
GIF = types['GIF'] = 'GIF'
ICO = types['ICO'] = 'ICO'
JPEG = types['JPEG'] = 'JPEG'
PNG = types['PNG'] = 'PNG'
TIFF = types['TIFF'] = 'TIFF'

image_fields = ['path', 'type', 'file_size', 'width', 'height']


class Image(collections.namedtuple('Image', image_fields)):

    def to_str_row(self):
        return ("%d\t%d\t%d\t%s\t%s" % (
            self.width,
            self.height,
            self.file_size,
            self.type,
            self.path.replace('\t', '\\t'),
        ))

    def to_str_row_verbose(self):
        return ("%d\t%d\t%d\t%s\t%s\t##%s" % (
            self.width,
            self.height,
            self.file_size,
            self.type,
            self.path.replace('\t', '\\t'),
            self))

    def to_str_json(self, indent=None):
        return json.dumps(self._asdict(), indent=indent, ensure_ascii=False)


def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct builtin modules
    """
    img = get_image_metadata(file_path)
    return (img.width, img.height)


def get_image_size_from_bytesio(input, size):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct builtin modules

    Args:
        input (io.IOBase): io object support read & seek
        size (int): size of buffer in byte
    """
    img = get_image_metadata_from_bytesio(input, size)
    return (img.width, img.height)


def get_image_metadata(file_path):
    """
    Return an `Image` object for a given img file content - no external
    dependencies except the os and struct builtin modules

    Args:
        file_path (str): path to an image file

    Returns:
        Image: (path, type, file_size, width, height)
    """
    size = os.path.getsize(file_path)

    # be explicit with open arguments - we need binary mode
    with io.open(file_path, "rb") as input:
        return get_image_metadata_from_bytesio(input, size, file_path)


def get_image_metadata_from_bytesio(input, size, file_path=None):
    """
    Return an `Image` object for a given img file content - no external
    dependencies except the os and struct builtin modules

    Args:
        input (io.IOBase): io object support read & seek
        size (int): size of buffer in byte
        file_path (str): path to an image file

    Returns:
        Image: (path, type, file_size, width, height)
    """
    height = -1
    width = -1
    data = input.read(26)
    msg = " raised while trying to decode as JPEG."

    if (size >= 10) and data[:6] in (b'GIF87a', b'GIF89a'):
        # GIFs
        imgtype = GIF
        w, h = struct.unpack("<HH", data[6:10])
        width = int(w)
        height = int(h)
    elif ((size >= 24) and data.startswith(b'\211PNG\r\n\032\n')
          and (data[12:16] == b'IHDR')):
        # PNGs
        imgtype = PNG
        w, h = struct.unpack(">LL", data[16:24])
        width = int(w)
        height = int(h)
    elif (size >= 16) and data.startswith(b'\211PNG\r\n\032\n'):
        # older PNGs
        imgtype = PNG
        w, h = struct.unpack(">LL", data[8:16])
        width = int(w)
        height = int(h)
    elif (size >= 2) and data.startswith(b'\377\330'):
        # JPEG
        imgtype = JPEG
        input.seek(0)
        input.read(2)
        b = input.read(1)
        try:
            while (b and ord(b) != 0xDA):
                while (ord(b) != 0xFF):
                    b = input.read(1)
                while (ord(b) == 0xFF):
                    b = input.read(1)
                if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                    input.read(3)
                    h, w = struct.unpack(">HH", input.read(4))
                    break
                else:
                    input.read(
                        int(struct.unpack(">H", input.read(2))[0]) - 2)
                b = input.read(1)
            width = int(w)
            height = int(h)
        except struct.error:
            raise UnknownImageFormat("StructError" + msg)
        except ValueError:
            raise UnknownImageFormat("ValueError" + msg)
        except Exception as e:
            raise UnknownImageFormat(e.__class__.__name__ + msg)
    elif (size >= 26) and data.startswith(b'BM'):
        # BMP
        imgtype = 'BMP'
        headersize = struct.unpack("<I", data[14:18])[0]
        if headersize == 12:
            w, h = struct.unpack("<HH", data[18:22])
            width = int(w)
            height = int(h)
        elif headersize >= 40:
            w, h = struct.unpack("<ii", data[18:26])
            width = int(w)
            # as h is negative when stored upside down
            height = abs(int(h))
        else:
            raise UnknownImageFormat(
                "Unkown DIB header size:" +
                str(headersize))
    elif (size >= 8) and data[:4] in (b"II\052\000", b"MM\000\052"):
        # Standard TIFF, big- or little-endian
        # BigTIFF and other different but TIFF-like formats are not
        # supported currently
        imgtype = TIFF
        byteOrder = data[:2]
        boChar = ">" if byteOrder == "MM" else "<"
        # maps TIFF type id to size (in bytes)
        # and python format char for struct
        tiffTypes = {
            1: (1, boChar + "B"),  # BYTE
            2: (1, boChar + "c"),  # ASCII
            3: (2, boChar + "H"),  # SHORT
            4: (4, boChar + "L"),  # LONG
            5: (8, boChar + "LL"),  # RATIONAL
            6: (1, boChar + "b"),  # SBYTE
            7: (1, boChar + "c"),  # UNDEFINED
            8: (2, boChar + "h"),  # SSHORT
            9: (4, boChar + "l"),  # SLONG
            10: (8, boChar + "ll"),  # SRATIONAL
            11: (4, boChar + "f"),  # FLOAT
            12: (8, boChar + "d")  # DOUBLE
        }
        ifdOffset = struct.unpack(boChar + "L", data[4:8])[0]
        try:
            countSize = 2
            input.seek(ifdOffset)
            ec = input.read(countSize)
            ifdEntryCount = struct.unpack(boChar + "H", ec)[0]
            # 2 bytes: TagId + 2 bytes: type + 4 bytes: count of values + 4
            # bytes: value offset
            ifdEntrySize = 12
            for i in range(ifdEntryCount):
                entryOffset = ifdOffset + countSize + i * ifdEntrySize
                input.seek(entryOffset)
                tag = input.read(2)
                tag = struct.unpack(boChar + "H", tag)[0]
                if (tag == 256 or tag == 257):
                    # if type indicates that value fits into 4 bytes, value
                    # offset is not an offset but value itself
                    type = input.read(2)
                    type = struct.unpack(boChar + "H", type)[0]
                    if type not in tiffTypes:
                        raise UnknownImageFormat(
                            "Unkown TIFF field type:" +
                            str(type))
                    typeSize = tiffTypes[type][0]
                    typeChar = tiffTypes[type][1]
                    input.seek(entryOffset + 8)
                    value = input.read(typeSize)
                    value = int(struct.unpack(typeChar, value)[0])
                    if tag == 256:
                        width = value
                    else:
                        height = value
                if width > -1 and height > -1:
                    break
        except Exception as e:
            raise UnknownImageFormat(str(e))
    elif size >= 2:
        # see http://en.wikipedia.org/wiki/ICO_(file_format)
        imgtype = 'ICO'
        input.seek(0)
        reserved = input.read(2)
        if 0 != struct.unpack("<H", reserved)[0]:
            raise UnknownImageFormat(FILE_UNKNOWN)
        format = input.read(2)
        assert 1 == struct.unpack("<H", format)[0]
        num = input.read(2)
        num = struct.unpack("<H", num)[0]
        if num > 1:
            import warnings
            warnings.warn("ICO File contains more than one image")
        # http://msdn.microsoft.com/en-us/library/ms997538.aspx
        w = input.read(1)
        h = input.read(1)
        width = ord(w)
        height = ord(h)
    else:
        raise UnknownImageFormat(FILE_UNKNOWN)

    return Image(path=file_path,
                 type=imgtype,
                 file_size=size,
                 width=width,
                 height=height)


import unittest


class Test_get_image_size(unittest.TestCase):
    data = [{
        'path': 'lookmanodeps.png',
        'width': 251,
        'height': 208,
        'file_size': 22228,
        'type': 'PNG'}]

    def setUp(self):
        pass

    def test_get_image_size_from_bytesio(self):
        img = self.data[0]
        p = img['path']
        with io.open(p, 'rb') as fp:
            b = fp.read()
        fp = io.BytesIO(b)
        sz = len(b)
        output = get_image_size_from_bytesio(fp, sz)
        self.assertTrue(output)
        self.assertEqual(output,
                         (img['width'],
                          img['height']))

    def test_get_image_metadata_from_bytesio(self):
        img = self.data[0]
        p = img['path']
        with io.open(p, 'rb') as fp:
            b = fp.read()
        fp = io.BytesIO(b)
        sz = len(b)
        output = get_image_metadata_from_bytesio(fp, sz)
        self.assertTrue(output)
        for field in image_fields:
            self.assertEqual(getattr(output, field), None if field == 'path' else img[field])

    def test_get_image_metadata(self):
        img = self.data[0]
        output = get_image_metadata(img['path'])
        self.assertTrue(output)
        for field in image_fields:
            self.assertEqual(getattr(output, field), img[field])

    def test_get_image_metadata__ENOENT_OSError(self):
        with self.assertRaises(OSError):
            get_image_metadata('THIS_DOES_NOT_EXIST')

    def test_get_image_metadata__not_an_image_UnknownImageFormat(self):
        with self.assertRaises(UnknownImageFormat):
            get_image_metadata('README.rst')

    def test_get_image_size(self):
        img = self.data[0]
        output = get_image_size(img['path'])
        self.assertTrue(output)
        self.assertEqual(output,
                         (img['width'],
                          img['height']))

    def tearDown(self):
        pass


def main(argv=None):
    """
    Print image metadata fields for the given file path.

    Keyword Arguments:
        argv (list): commandline arguments (e.g. sys.argv[1:])
    Returns:
        int: zero for OK
    """
    import logging
    import optparse
    import sys

    prs = optparse.OptionParser(
        usage="%prog [-v|--verbose] [--json|--json-indent] <path0> [<pathN>]",
        description="Print metadata for the given image paths "
                    "(without image library bindings).")

    prs.add_option('--json',
                   dest='json',
                   action='store_true')
    prs.add_option('--json-indent',
                   dest='json_indent',
                   action='store_true')

    prs.add_option('-v', '--verbose',
                   dest='verbose',
                   action='store_true', )
    prs.add_option('-q', '--quiet',
                   dest='quiet',
                   action='store_true', )
    prs.add_option('-t', '--test',
                   dest='run_tests',
                   action='store_true', )

    argv = list(argv) if argv is not None else sys.argv[1:]
    (opts, args) = prs.parse_args(args=argv)
    loglevel = logging.INFO
    if opts.verbose:
        loglevel = logging.DEBUG
    elif opts.quiet:
        loglevel = logging.ERROR
    logging.basicConfig(level=loglevel)
    log = logging.getLogger()
    log.debug('argv: %r', argv)
    log.debug('opts: %r', opts)
    log.debug('args: %r', args)

    if opts.run_tests:
        import sys
        sys.argv = [sys.argv[0]] + args
        import unittest
        return unittest.main()

    output_func = Image.to_str_row
    if opts.json_indent:
        import functools
        output_func = functools.partial(Image.to_str_json, indent=2)
    elif opts.json:
        output_func = Image.to_str_json
    elif opts.verbose:
        output_func = Image.to_str_row_verbose

    EX_OK = 0
    EX_NOT_OK = 2

    if len(args) < 1:
        prs.print_help()
        print('')
        prs.error("You must specify one or more paths to image files")

    errors = []
    for path_arg in args:
        try:
            img = get_image_metadata(path_arg)
            print(output_func(img))
        except KeyboardInterrupt:
            raise
        except OSError as e:
            log.error((path_arg, e))
            errors.append((path_arg, e))
        except Exception as e:
            log.exception(e)
            errors.append((path_arg, e))
            pass
    if len(errors):
        import pprint
        print("ERRORS", file=sys.stderr)
        print("======", file=sys.stderr)
        print(pprint.pformat(errors, indent=2), file=sys.stderr)
        return EX_NOT_OK
    return EX_OK


is_window_shown = False
display_lock = threading.Lock()
current_img = None
update_event = threading.Event()

def update_image(img, name):
    global current_img
    with display_lock:
        current_img = (img, name)
        update_event.set()

def display_image_in_thread():
    global is_window_shown

    def display_img():
        global current_img
        while True:
            update_event.wait()
            with display_lock:
                if current_img:
                    img, name = current_img
                    cv2.imshow(name, img)
                    current_img = None
                    update_event.clear()
                if cv2.waitKey(1) & 0xFF == 27:  # Esc key to stop
                    cv2.destroyAllWindows()
                    print('\nESC pressed, stopping')
                    break

    if not is_window_shown:
        is_window_shown = True
        threading.Thread(target=display_img, daemon=True).start()


def show_img(img, name='AI Toolkit'):
    img = np.clip(img, 0, 255).astype(np.uint8)
    update_image(img[:, :, ::-1], name)
    if not is_window_shown:
        display_image_in_thread()


def show_tensors(imgs: torch.Tensor, name='AI Toolkit'):
    if len(imgs.shape) == 4:
        img_list = torch.chunk(imgs, imgs.shape[0], dim=0)
    else:
        img_list = [imgs]

    img = torch.cat(img_list, dim=3)
    img = img / 2 + 0.5
    img_numpy = img.to(torch.float32).detach().cpu().numpy()
    img_numpy = np.clip(img_numpy, 0, 1) * 255
    img_numpy = img_numpy.transpose(0, 2, 3, 1)
    img_numpy = img_numpy.astype(np.uint8)

    show_img(img_numpy[0], name=name)
    
def save_tensors(imgs: torch.Tensor, path='output.png'):
    if len(imgs.shape) == 5 and imgs.shape[0] == 1:
        imgs = imgs.squeeze(0)
    if len(imgs.shape) == 4:
        img_list = torch.chunk(imgs, imgs.shape[0], dim=0)
    else:
        img_list = [imgs]

    img = torch.cat(img_list, dim=3)
    img = img / 2 + 0.5
    img_numpy = img.to(torch.float32).detach().cpu().numpy()
    img_numpy = np.clip(img_numpy, 0, 1) * 255
    img_numpy = img_numpy.transpose(0, 2, 3, 1)
    img_numpy = img_numpy.astype(np.uint8)
    # concat images to one
    img_numpy = np.concatenate(img_numpy, axis=1)
    # conver to pil
    img_pil = PILImage.fromarray(img_numpy)
    img_pil.save(path)

def show_latents(latents: torch.Tensor, vae: 'AutoencoderTiny', name='AI Toolkit'):
    if vae.device == 'cpu':
        vae.to(latents.device)
    latents = latents / vae.config['scaling_factor']
    imgs = vae.decode(latents).sample
    show_tensors(imgs, name=name)


def on_exit():
    if is_window_shown:
        cv2.destroyAllWindows()


def reduce_contrast(tensor, factor):
    # Ensure factor is between 0 and 1
    factor = max(0, min(factor, 1))

    # Calculate the mean of the tensor
    mean = torch.mean(tensor)

    # Reduce contrast
    adjusted_tensor = (tensor - mean) * factor + mean

    # Clip values to ensure they stay within -1 to 1 range
    return torch.clamp(adjusted_tensor, -1.0, 1.0)

atexit.register(on_exit)

if __name__ == "__main__":
    import sys

    sys.exit(main(argv=sys.argv[1:]))
