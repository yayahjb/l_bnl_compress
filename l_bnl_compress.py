'''l_bnl_compress.py, lossy, but not lossy, compresss,
       a script to apply lossy compression to HDF5 MX image files.

  (C) Copyright 16 March 2025 Herbert J. Bernstein
  Portions suggested by claude.ai from Anthropic
  You may redistribute l_bnl_compress.py under GPL2 or LGPL2 
 
usage: l_bnl_compress.py [-h] [-1 FIRST_IMAGE] [-b BIN_RANGE] [-c COMPRESSION] [-d DATA_BLOCK_SIZE] \
                         [-H HCOMP_SCALE] [-i INFILE] [-J J2K_TARGET_COMPRESSION_RATIO] \
                         [-l COMPRESSION_LEVEL] [-m OUT_MASTER] [-N LAST_IMAGE] [-o OUT_FILE] \
                         [-q OUT_SQUASH] [-s SUM_RANGE] [-v]

Bin and sum images from a range

options:
  -h, --help            show this help message and exit
  -1 FIRST_IMAGE, --first_image FIRST_IMAGE
                        first selected image counting from 1
  -b BIN_RANGE, --bin BIN_RANGE
                        an integer image binning range (1 ...) to apply to each selected image
  -c COMPRESSION, --compression COMPRESSION
                        optional compression, bslz4, bszstd, bshuf, or zstd
  -d DATA_BLOCK_SIZE, --data_block_size DATA_BLOCK_SIZE
                        data block size in images for out_file
  -H HCOMP_SCALE, --Hcompress HCOMP_SCALE
                        Hcompress scale compression, immediately followed by decompression
  -i INFILE, --infile INFILE
                        the input hdf5 file to read images from
  -J J2K_TARGET_COMPRESSION_RATIO, --J2K J2K_TARGET_COMPRESSION_RATIO
                        JPEG-2000 target compression ratio, immediately followed by decompression
  -l COMPRESSION_LEVEL, --compression_level COMPRESSION_LEVEL
                        optional compression level for bszstd or zstd
  -m OUT_MASTER, --out_master OUT_MASTER
                        the output hdf5 master to which to write metadata, defaults to OUT_FILE_MASTER
                        if not given, out given as out_file
  -N LAST_IMAGE, --last_image LAST_IMAGE
                        last selected image counting from 1
  -o OUT_FILE, --out_file OUT_FILE
                        the output hdf5 data file out_file_?????? with an .h5 extension are files to which to write images
  -q OUT_SQUASH, --out_squash OUT_SQUASH
                        an optional hdf5 data file out_squash_?????? with an .h5 extension are optional files to which
                        raw j2k or hcomp files paralleling OUT_FILE are written, defaults to OUT_FILE_SQUASH
                        if given as out_file 
  -s SUM_RANGE, --sum SUM_RANGE
                        an integer image summing range (1 ...) to apply to the selected images
  -u SIZE ,--uint SIZE
                        clip the output above 0 and limit to 2 byte or 4 byte integers 
  -v, --verbose         provide addtional information

  -V, --version         report the version and build_date



'''

import sys
import os
import argparse
import numpy as np
import skimage as ski
import h5py
import tifffile
from astropy.io import fits
from astropy.io.fits.hdu.compressed import COMPRESSION_TYPES
import glymur
import hdf5plugin
import tempfile
import numcodecs
from astropy.io.fits.hdu.compressed._codecs import HCompress1
from io import BytesIO

def compress_HCarray(input_array, satval=32767, scale=16):
    """
    Compress a numpy int16 array using HCompress with lossy compression.
    
    Parameters:
    -----------
    input_array : numpy.ndarray
        Input array with numpy.int16 (np.int16) dtype
    satval : int, optional
        Saturation value, default is 32767 (max value for signed 16-bit)
    scale : int, optional
        Compression scale factor (higher for more compression), default is 16
        Common values are 12 or 16 for lossy compression
    
    Returns:
    --------
    compressed_data : bytes
        Compressed data as bytes
    shape : tuple
        Original shape of the array for later decompression
    """
    # Ensure input is np.int16
    if input_array.dtype != np.int16:
        input_array = input_array.astype(np.int16)
    
    # Clip values between 0 and satval
    clipped_array = np.clip(input_array, 0, satval)
    
    # Get original shape for decompression
    original_shape = clipped_array.shape
    
    # Create a compressed HDU using HCompress
    comp_hdu = fits.CompImageHDU(data=clipped_array, 
                                 compression_type='HCOMPRESS_1',
                                 hcomp_scale=scale)
    
    # Write to BytesIO object
    buffer = fits.HDUList([fits.PrimaryHDU(), comp_hdu])
    bio = BytesIO()
    buffer.writeto(bio)
    bio.seek(0)
    
    # Read the compressed data back and store the entire compressed HDU
    with fits.open(bio) as hdul:
        # Store the entire FITS file as bytes for later decompression
        bio.seek(0)  # Reset to the beginning
        fits_bytes = bio.read()
    
    # Return the entire FITS file, original shape
    return fits_bytes, original_shape

def decompress_HCarray(fits_bytes, original_shape, scale=16):
    """
    Decompress HCompressed data back to a numpy int16 array.
    
    Parameters:
    -----------
    fits_bytes : bytes
        Complete FITS file as bytes from compress_array
    original_shape : tuple
        Original shape of the array
    scale : int, optional
        Compression scale factor used for compression, default is 16
    
    Returns:
    --------
    decompressed_array : numpy.ndarray
        Decompressed array with the same shape as the original, as np.int16
    """
    # Create a BytesIO object from the FITS bytes
    bio = BytesIO(fits_bytes)
    
    # Open the FITS file from memory
    with fits.open(bio) as hdul:
        # Extract the decompressed data
        decompressed_array = hdul[1].data.copy()
    
    # Ensure output is np.int16
    if decompressed_array.dtype != np.int16:
        decompressed_array = decompressed_array.astype(np.int16)
    
    return decompressed_array


version = "1.1.1"
version_date = "10Mar25"
xnt=int(1)

def ntstr(xstr):
    return str(xstr)

def ntstrdt(str):
    return h5py.string_dtype(encoding='utf-8',length=len(str)+xnt)

def conv_pixel_mask(old_mask,bin_range):
    ''' conv_pixel_mask -- returns a new pixel_mask
    array, adjusting for binning
    '''

    if bin_range < 2 :
        return np.asarray(old_mask,dtype='u4')
    old_shape=old_mask.shape
    if len(old_shape) != 2:
        print('l_bnl_compress.py: invalid mask shape for 2D binning')
        return None
    sy=old_shape[0]
    nsy=int(sy+bin_range-1)//bin_range
    ymargin=0
    if nsy*bin_range > sy:
        ymargin=bin_range-(sy%bin_range)
    sx=old_shape[1]
    nsx=int(sx+bin_range-1)//bin_range
    xmargin=0
    if nsx*bin_range > sx:
        xmargin=bin_range-(sx%bin_range)
    if ((xmargin > 0) or (ymargin > 0)):
        old_mask_rev=np.pad(np.asarray(old_mask,dtype='u4'),((0,ymargin),(0,xmargin)),\
        'constant',constant_values=((0,0),(0,0)))
    else:
        old_mask_rev=np.asaary(old_mask,dtype='u4')
    new_mask=np.zeros((nsy,nsx),dtype='u4')
    for iy in range(0,sy,bin_range):
        for ix in range(0,sx,bin_range):
            for iyy in range(0,bin_range):
                for ixx in range(0,bin_range):
                    if ix+ixx < sx and iy+iyy < sy:
                        if old_mask_rev[iy+iyy,ix+ixx] != 0:
                            new_mask[iy//bin_range,ix//bin_range] = new_mask[iy//bin_range,ix//bin_range]| \
                            old_mask_rev[iy+iyy,ix+ixx]
    return new_mask


def conv_image_to_block_offset(img,npb):
    ''' conv_image_to_block_offset(img,npb)

    convert an image number, img, counting from 1,
    given the number of images per block, npb,
    into a 2-tuple consisting of the block
    number, counting from zero, and an image
    nuber within that block counting from zero.
    '''

    nblk=(img-1)//npb
    return (nblk,(int(img-1)%int(npb)))

def conv_image_shqpe(old_shape,bin_range):
    if len(old_shape) != 2:
        print('l_bnl_compress.py: invalid image shape for 2D binning')
        return None
    sy=old_shape[0]
    nsy=int(sy+bin_range-1)//bin_range
    ymargin=0
    if nsy*bin_range > sy:
        ymargin=bin_range-(sy%bin_range)
    sx=old_shape[1]
    nsx=int(sx+bin_range-1)//bin_range
    xmargin=0
    if nsx*bin_range > sx:
        xmargin=bin_range-(sx%bin_range)
    return ((sy+ymargin)//bin_range,(sx+xmargin)//bin_range)

def xfer_axis_attrs(dst,src):
    if src != None:
        mykeys=src.attrs.keys()
        if 'units' in mykeys:
            dst.attrs.create('units',ntstr(src.attrs['units']),\
            dtype=ntstrdt(src.attrs['units']))
        if 'depends_on' in mykeys:
            dst.attrs.create('depends_on',ntstr(src.attrs['depends_on']),\
            dtype=ntstrdt(src.attrs['depends_on']))
        if 'transformation_type' in mykeys:
            dst.attrs.create('transformation_type',ntstr(src.attrs['transformation_type']),\
            dtype=ntstrdt(src.attrs['transformation_type']))
        if 'vector' in mykeys:
            dst.attrs.create('vector',src.attrs['vector'],\
            dtype=src.attrs['vector'].dtype)
        if 'offset' in mykeys:
            dst.attrs.create('offset',src.attrs['offset'],\
            dtype=src.attrs['offset'].dtype)



def bin_array(input_array, nbin, satval):
    """
    Bin a 2D numpy array into blocks of nbin x nbin pixels.
    
    Parameters:
    -----------
    input_array : numpy.ndarray
        2D input array to be binned
    nbin : int
        Size of the binning block (nbin x nbin)
    satval : int
        Maximum value to clip the summed block values
    
    Returns:
    --------
    numpy.ndarray
        Binned array with reduced dimensions
    """
    # Determine padded dimensions
    height, width = input_array.shape
    padded_height = int(np.ceil(height / nbin) * nbin)
    padded_width = int(np.ceil(width / nbin) * nbin)
    
    # Create padded array initialized with zeros
    padded_array = np.zeros((padded_height, padded_width), dtype=input_array.dtype)
    
    # Copy original array into padded array
    padded_array[:height, :width] = input_array
    
    # Reshape and sum with clipping
    reshaped = padded_array.reshape(
        padded_height // nbin, nbin, 
        padded_width // nbin, nbin
    )
    
    # Sum each nbin x nbin block and clip
    binned_array = np.clip(
        reshaped.sum(axis=(1, 3)), 
        0, 
        satval
    )
    
    return binned_array



def bin(old_image,bin_range,satval):
    ''' bin(old_image,bin_range,satval)

    convert an image in old_image to a returned u2 numpy array by binning
    the pixels in old_image in by summing bin_range by bin_range rectanglar 
    blocks, clipping values between 0 and satval.  If bin_range does not divide
    the original dimensions exactly, the old_image is padded with zeros.
    '''
    s=old_image.shape
    if len(s) != 2:
        print('l_bnl_compress.py: invalid image shape for 2D binning')
        return None
    new_image = np.maximum(old_image,0).astype(np.uint16)
    if bin_range < 2:
        return new_image.clip(0,satval)
    sy=s[0]
    nsy=int(sy+bin_range-1)//bin_range
    ymargin=0
    if nsy*bin_range > sy:
        ymargin=bin_range-(sy%bin_range)
    sx=s[1]
    nsx=int(sx+bin_range-1)//bin_range
    xmargin=0
    if nsx*bin_range > sx:
        xmargin=bin_range-(sx%bin_range)
    if ((xmargin > 0) or (ymargin > 0)):
        new_image=np.clip(np.pad(np.asarray(new_image,dtype='u2'),((0,ymargin),(0,xmargin)),'constant',constant_values=((0,0),(0,0))),0,satval)
    else:
        new_image=np.clip(np.asarray(new_image,dtype='u2'),0,satval)
    new_image=(np.asarray(new_image,dtype='u2')).clip(0,satval)
    new_image=np.round(ski.measure.block_reduce(new_image,(bin_range,bin_range),np.sum))
    new_image=np.asarray(np.clip(new_image,0,satval),dtype='u2')
    if args['verbose']==True:
        print('converted with bin ',s,' to ',new_image.shape,' with dtype u2')
    return new_image


parser = argparse.ArgumentParser(description='Bin and sum images from a range')
parser.add_argument('-1','--first_image', dest='first_image', type=int, nargs='?', const=1, default=1,
   help= 'first selected image counting from 1, defaults to 1')
parser.add_argument('-b','--bin', dest='bin_range', type=int, nargs='?', const=1, default=1,
   help= 'an integer image binning range (1 ...) to apply to each selected image, defaults to 1') 
parser.add_argument('-c','--compression', dest='compression', nargs='?', const='zstd', default='zstd',
   help= 'optional compression, bslz4, bszstd,  bshuf, or zstd, defaults to zstd')
parser.add_argument('-d','--data_block_size', dest='data_block_size', type=int, nargs='?', const=100, default=100,
   help= 'data block size in images for out_file, defaults to 100')
parser.add_argument('-H','--Hcompress', dest='hcomp_scale', type=int,
   help= 'Hcompress scale compression, immediately followed by decompression')
parser.add_argument('-i','--infile',dest='infile',
   help= 'the input hdf5 file to read images from')
parser.add_argument('-J','--J2K', dest='j2k_target_compression_ratio', type=int,
   help= 'JPEG-2000 target compression ratio, immediately followed by decompression')
parser.add_argument('-l','--compression_level', dest='compression_level', type=int,
   help= 'optional compression level for bszstd or zstd')
parser.add_argument('-m','--out_master',dest='out_master',
   help= 'the output hdf5 master to which to write metadata')
parser.add_argument('-N','--last_image', dest='last_image', type=int,
   help= 'last selected image counting from 1, defaults to number of images collected')
parser.add_argument('-o','--out_file',dest='out_file',default='out_data',
   help= 'the output hdf5 data file out_file_?????? with an .h5 extension are files to which to write images')
parser.add_argument('-q','--out_squash',dest='out_squash',
   help= 'the output hdf5 data file out_squash_?????? with an .h5 extension are optional files to which to write raw j2k or hcomp images')
parser.add_argument('-s','--sum', dest='sum_range', type=int, nargs='?', const=1, default=1,
   help= 'an integer image summing range (1 ...) to apply to the selected images, defaults to 1')
parser.add_argument('-v','--verbose',dest='verbose',action='store_true',
   help= 'provide addtional information')
parser.add_argument('-u','--uint',dest='unit",type=int,nargs='?',const=2, default=2,
   help= 'clip the output above 0 and limit to 2 byte or 4 byte integers',
parser.add_argument('-V','--version',dest='report_version',action='store_true',
   help= 'report version and version_date')
args = vars(parser.parse_args())

#h5py._hl.filters._COMP_FILTERS['blosc']    =32001
#h5py._hl.filters._COMP_FILTERS['lz4']      =32004
#h5py._hl.filters._COMP_FILTERS['bshuf']    =32008
#h5py._hl.filters._COMP_FILTERS['zfp']      =32013
#h5py._hl.filters._COMP_FILTERS['zstd']     =32015
#h5py._hl.filters._COMP_FILTERS['sz']       =32017
#h5py._hl.filters._COMP_FILTERS['fcidecomp']=32018
#h5py._hl.filters._COMP_FILTERS['jpeg']     =32019
#h5py._hl.filters._COMP_FILTERS['sz3']      =32024
#h5py._hl.filters._COMP_FILTERS['blosc2']   =32026
#h5py._hl.filters._COMP_FILTERS['j2k']      =32029
#h5py._hl.filters._COMP_FILTERS['hcomp']    =32030

if args['report_version'] == True:
    print('l_bnl_compress-'+version+'-'+version_date)

if args['verbose'] == True:
    print(args)
    #print(h5py._hl.filters._COMP_FILTERS)

try:
    fin = h5py.File(args['infile'], 'r')
except:
    print('l_bnl_compress.py: infile not specified')
    sys.exit(-1)

try:
    top_definition=fin['entry']['definition']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/definition: ', top_definition)
        print('                   entry/definition[()]: ', top_definition[()])
except:
    print('l_bnl_compress.py: entry/definition not found')
    top_definition = None

try:
    detector=fin['entry']['instrument']['detector']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector: ', detector)
except:
    print('l_bnl_compress.py: detector not found')
    detector = None

try:
    description=detector['description']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/description: ', description)
        print('                   detector/description[()]: ', description[()])
except:
    print('l_bnl_compress.py: detector/description not found')
    description = None

try:
    detector_number=detector['detector_number']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector_number: ', detector_number)
        print('                   detector_number[()]: ', detector_number[()])
except:
    print('l_bnl_compress.py: detector/detector_number not found')
    detector_number=None

try:
    depends_on=detector['depends_on']
    if args['verbose'] == True:
        print('l_bnl_compress.py: depends_on: ', depends_on)
        print('                   depends_on[()]: ', depends_on[()])
except:
    print('l_bnl_compress.py: detector/depends_on not found')
    depends_on=None

try:
    bit_depth_image=detector['bit_depth_image']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/bit_depth_image: ', bit_depth_image)
        print('                   detector/bit_depth_image[()]: ', bit_depth_image[()])
except:
    print('l_bnl_compress.py: detector/bit_depth_image not found')
    bit_depth_image=None

try:
    bit_depth_readout=detector['bit_depth_readout']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/bit_depth_readout: ', bit_depth_readout)
        print('                   detector/bit_depth_readout[()]: ', bit_depth_readout[()])
except:
    print('l_bnl_compress.py: detector/bit_depth_readout not found')
    bit_depth_readout = None

try:
    thickness=detector['sensor_thickness']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/sensor_thickness: ', thickness)
        print('                   detector/sensor_thickness[()]: ', thickness[()])
except:
    print('l_bnl_compress.py: detector/sensor_thickness not found')
    thickness='unknown'

pixel_mask=None
try:
    pixel_mask=detector['pixel_mask']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/pixel_mask: ', pixel_mask)
        print('                 detector/pixel_mask[()]: ', pixel_mask[()])
except:
    print('l_bnl_compress.py: detector/pixel_mask not found')
    pixel_mask=None

try:
    beamx=detector['beam_center_x']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/beam_center_x: ', beamx)
        print('                 detector/beam_center_x[()]: ', beamx[()])
except:
    print('l_bnl_compress.py: detector/beam_center_x not found')
    fin.close()
    sys.exit(-1)

try:
    beamy=detector['beam_center_y']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/beam_center_y: ', beamy)
        print('                 detector/beam_center_y[()]: ', beamy[()])
except:
    print('l_bnl_compress.py: detector/beam_center_y not found')
    fin.close()
    sys.exit(-1)

try:
    count_time=detector['count_time']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/count_time: ', count_time)
        print('                 detector/count_time[()]: ', count_time[()])
except:
    print('l_bnl_compress.py: detector/count_time not found')
    fin.close()
    sys.exit(-1)

try:
    countrate_correction_applied=detector['countrate_correction_applied']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/countrate_correction_applied: ', countrate_correction_applied)
        print('                 detector/countrate_correction_applied[()]: ', countrate_correction_applied[()])
except:
    print('l_bnl_compress.py: detector/countrate_correction_applied not found')
    countrate_correction_applied=None

try:
    distance=detector['distance']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/distance: ', distance)
        print('                 detector/distance[()]: ', distance[()])
except:
    print('l_bnl_compress.py: detector/detector_distance not found')
    try:
        distance=detector['detector_distance']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/detector_distance: ', distance)
            print('                 detector/detector_distance[()]: ', distance[()])
    except:
        print('l_bnl_compress.py: detector/detector_distance not found')
        fin.close()
        sys.exit(-1)

try:
    frame_time=detector['frame_time']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/frame_time: ', frame_time)
        print('                   detector/frame_time[()]: ', frame_time[()])
        print('                   detector/frame_time.attrs.keys(): ',frame_time.attrs.keys())
except:
    print('l_bnl_compress.py: detector/frame_time not found')
    fin.close()
    sys.exit(-1)

satval = 32767
satval_not_found = True
try:
    saturation_value = detector['saturation_value']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/saturation_value: ', saturation_value)
        print('                   detector/saturation_value[()]: ', saturation_value[()])
    satval = saturation_value[()]
    satval_not_found = False 
except:
    print('l_bnl_compress.py: detector/saturation_value not found')
    saturation_value = None

try:
    pixelsizex=detector['x_pixel_size']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/x_pixel_size: ', pixelsizex)
        print('                 detector/x_pixel_size[()]: ', pixelsizex[()])
except:
    print('l_bnl_compress.py: detector/x_pixel_size not found')
    fin.close()
    sys.exit(-1)

try:
    pixelsizey=detector['y_pixel_size']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/y_pixel_size: ', pixelsizey)
        print('                 detector/y_pixel_size[()]: ', pixelsizey[()])
except:
    print('l_bnl_compress.py: detector/y_pixel_size not found')
    fin.close()
    sys.exit(-1)

try:
    detectorSpecific=fin['entry']['instrument']['detector']['detectorSpecific']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific: ', detectorSpecific)
except:
    print('l_bnl_compress.py: detectorSpecific not found')
    fin.close()
    sys.exit(-1)

try:
    xpixels=detectorSpecific['x_pixels_in_detector']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/x_pixels_in_detector: ', xpixels)
        print('                 detectorSpecific/x_pixels_in_detector[()]: ', xpixels[()])
except:
    print('l_bnl_compress.py: detectorSpecific/x_pixels_in_detector not found')
    fin.close()
    sys.exit(-1)

try:
    ypixels=detectorSpecific['y_pixels_in_detector']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/y_pixels_in_detector: ', ypixels)
        print('l_bnl_compress.py: detectorSpecific/y_pixels_in_detector: ', ypixels[()])
except:
    print('l_bnl_compress.py: detectorSpecific/y_pixels_in_detector not found')
    fin.close()
    sys.exit(-1)

try:
    xnimages=detectorSpecific['nimages']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/nimages: ', xnimages)
        print('                 detectorSpecific/nimages[()]: ', xnimages[()])
except:
    print('l_bnl_compress.py: detectorSpecific/nimages not found')
    fin.close()
    sys.exit(-1)

try:
    xntrigger=detectorSpecific['ntrigger']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/ntrigger: ', xntrigger)
        print('                 detectorSpecific/ntrigger[()]: ', xntrigger[()])
except:
    print('l_bnl_compress.py: detectorSpecific/ntrigger not found')
    fin.close()
    sys.exit(-1)

nimages=int(xnimages[()])
ntrigger=int(xntrigger[()])
if args['verbose'] == True:     
    print('nimages: ',nimages)
    print('ntrigger: ',ntrigger)
if nimages == 1:   
    nimages = ntrigger
    print('l_bnl_compress.py: warning: settng nimages to ',nimages,' from ntrigger')
if args['last_image'] == None:
    args['last_image'] =  nimages 

try:
    software_version=detectorSpecific['software_version']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/software_version: ', software_version)
        print('                 detectorSpecific/software_version[()]: ', software_version[()])
except:
    print('l_bnl_compress.py: detectorSpecific/software_version not found')
    fin.close()
    sys.exit(-1)

try:
    countrate_correction_count_cutoff = detectorSpecific['countrate_correction_count_cutoff']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/countrate_correction_count_cutoff: ', countrate_correction_count_cutoff)
        print('                   detectorSpecific/countrate_correction_count_cutoff[()]: ', countrate_correction_count_cutoff[()])
    if satval_not_found:
        satval = countrate_correction_count_cutoff[()]
        satval_not_found = False 

except:
    print('l_bnl_compress.py: detectorSpecific/countrate_correction_count_cutoff not found')
    countrate_correction_count_cutoff = None

try:
    dS_saturation_value = detectorSpecific['saturation_value']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/saturation_value: ', dS_saturation_value)
        print('                   detectorSpecific/saturation_value[()]: ', dS_saturation_value[()])
    if satval_not_found:
        satval = dS_saturation_value[()]
        satval_not_found = False 
except:
     print('l_bnl_compress.py: detectorSpecific/saturation_value not found')
     dS_saturation_value = None  

try:
    mod0_countrate_cutoff = detectorSpecific['detectorModule_000']['countrate_correction_count_cutoff']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/detectorModule_000/countrate_correction_count_cutoff: ',mod0_countrate_cutoff)
        print('                   detectorSpecific/detectorModule_000countrate_correction_count_cutoff[()]: ',  mod0_countrate_cutoff[()])
        print('                  *** use of this dataset is deprecated ***')
    if satval_not_found:
        satval = mod0_countrate_cutoff[()]
        satval_not_found = False
except:
    print('l_bnl_compress.py: detectorSpecific/detectorModule_000/countrate_correction_count_cutoff not found')
    if satval_not_found:
        print('l_bnl_compress.py: ...count_cutoff not found, using 32765')
        satval=32765
    mod0_countrate_cutoff = None

dS_pixel_mask = None
try:
    dS_pixel_mask=detectorSpecific['pixel_mask']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detectorSpecific/pixel_mask: ', dS_pixel_mask)
        print('                   detectorSpecific/pixel_mask[()]: ', dS_pixel_mask[()])
except:
    print('l_bnl_compress.py: detectorSpecific/pixel_mask not found')
    dS_pixel_mask=None

det_gon = None
det_gon_two_theta = None
det_gon_two_theta_end = None
det_gon_two_theta_range_average = None
det_gon_two_theta_range_total = None
try:
    det_gon = detector['goniometer']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/goniometer: ', det_gon)
        print('                   detector/goniometer[()]: ', det_gon[()])
    try:
        det_gon_two_theta = detector['goniometer']['two_theta']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/goniometer/two_theta: ', det_gon_two_theta)
            print('                   detector/goniometer/two_theta[()]: ', det_gon_two_theta[()])
    except:
        print('l_bnl_compress.py: detector/goniometer/two_theta not found')
        det_gon_translation = None
    try:
        det_gon_two_theta_end = detector['goniometer']['two_theta_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/goniometer/two_theta_end: ', det_gon_two_theta_end)
            print('                   detector/goniometer/two_theta_end[()]: ', det_gon_two_theta_end[()])
    except:
        print('l_bnl_compress.py: detector/goniometer/two_theta_end not found')
        det_gon_two_theta_end = None
    try:
        det_gon_two_theta_range_average = detector['goniometer']['two_theta_range_average']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/goniometer/two_theta_range_average: ',\
            det_gon_two_theta_range_average)
            print('                   detector/goniometer/two_theta_range_average[()]: ',\
            det_gon_two_theta_range_average[()])
    except:
        print('l_bnl_compress.py: detector/goniometer/two_theta_range_average not found')
        det_gon_range_average = None
    try:
        det_gon_two_theta_range_total = detector['goniometer']['two_theta_range_total']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/goniometer/two_theta_range_total: ',\
            det_gon_two_theta_range_total)
            print('                   detector/goniometer/two_theta_range_total[()]: ',\
            det_gon_two_theta_range_total[()])
    except:
        print('l_bnl_compress.py: detector/goniometer/two_theta_range_total not found')
        det_gon_two_theta_range_total = None
except:
    print('l_bnl_compress.py: detector/goniometer not found')
    det_gon = None



det_nxt = None
det_nxt_transation = None
det_nxt_two_theta = None
det_nxt_two_theta_end = None
det_nxt_two_theta_range_average = None
det_nxt_two_theta_range_total = None
try:
    det_nxt = detector['transformations']
    if args['verbose'] == True:
        print('l_bnl_compress.py: detector/transformations: ', det_nxt)
        print('                   detector/transformations[()]: ', det_nxt[()])
    try:
        det_nxt_translation = detector['transformations']['translation']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/transformations/translation: ', det_nxt_translation)
            print('                   detector/transformations/translation[()]: ', det_nxt_translation[()])
    except:
        print('l_bnl_compress.py: detector/transformations/translation not found')
        det_nxt_translation = None
    try:
        det_nxt_two_theta = detector['transformations']['two_theta']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/transformations/two_theta: ', det_nxt_two_theta)
            print('                   detector/transformations/two_theta[()]: ', det_nxt_two_theta[()])
    except:
        print('l_bnl_compress.py: detector/transformations/two_theta not found')
        det_nxt_translation = None
    try:
        det_nxt_two_theta_end = detector['transformations']['two_theta_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/transformations/two_theta_end: ', det_nxt_two_theta_end)
            print('                   detector/transformations/two_theta_end[()]: ', det_nxt_two_theta_end[()])
    except:
        print('l_bnl_compress.py: detector/transformations/two_theta_end not found')
        det_nxt_two_theta_end = None
    try:
        det_nxt_two_theta_range_average = detector['transformations']['two_theta_range_average']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/transformations/two_theta_range_average: ',\
            det_nxt_two_theta_range_average)
            print('                   detector/transformations/two_theta_range_average[()]: ',\
            det_nxt_two_theta_range_average[()])
    except:
        print('l_bnl_compress.py: detector/transformations/two_theta_range_average not found')
        det_nxt_range_average = None
    try:
        det_nxt_two_theta_range_total = detector['transformations']['two_theta_range_total']
        if args['verbose'] == True:
            print('l_bnl_compress.py: detector/transformations/two_theta_range_total: ',\
            det_nxt_two_theta_range_total)
            print('                   detector/transformations/two_theta_range_total[()]: ',\
            det_nxt_two_theta_range_total[()])
    except:
        print('l_bnl_compress.py: detector/transformations/two_theta_range_total not found')
        det_nxt_two_theta_range_total = None
except:
    print('l_bnl_compress.py: detector/transformations not found')
    det_nxt = None

best_wavelength = None
try:
    sample_wavelength=fin['entry']['sample']['beam']['incident_wavelength']
    if best_wavelength == None:
        best_wavelength = sample_wavelength
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/beam/incident_wavelength: ', sample_wavelength)
        print('                   entry/sample/beam/incident_wavelength[()]: ', sample_wavelength[()])
except:
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/beam/incident_wavelength not found')
    sample_wavelength = None

try:
    instrument_wavelength=fin['entry']['instrument']['beam']['wavelength']
    if best_wavelength == None:
        best_wavelength = instrument_wavelength
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/instrument/beam/wavelength: ', instrument_wavelength)
        print('                   entry/instrument/beam/wavelength[()]: ', instrument_wavelength[()])
except:
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/instrument/beam/wavelength not found')
    instrument_wavelength = None

try:
    monochromater_wavelength=fin['entry']['instrument']['monochromater']['wavelength']
    if best_wavelength == None:
        best_wavelength = monochromater_wavelength
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/instrument/monochromater/wavelength: ', monochromater_wavelength)
        print('                   entry/instrument/monochromater/wavelength[()]: ', monochromater_wavelength[()])
except:
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/instrument/monochromater/wavelength not found')
    monochromater_wavelength=None

try:
    beam_incident_wavelength=fin['entry']['instrument']['beam']['incident_wavelength']
    if best_wavelength == None:
        best_wavelength = beam_incident_wavelength
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/instrument/beam/incident_wavelength: ', beam_incident_wavelength)
        print('                 entry/instrument/beam/incident_wavelength[()]: ', beam_incident_wavelength[()])
except:
    print('l_bnl_compress.py:entry/instrument/beam/incident_wavelength not found')
    beam_incident_wavelength = None

if best_wavelength==None:
    print('l_bnl_compress.py: ... wavelength not found')
    fin.close()
    sys.exit(-1)


# find {chi, kappa, omega, phi, translation} in entry/sample/transformations/*
try:
    nxt_chi=fin['entry']['sample']['transformations']['chi']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/chi: ', nxt_chi)
        print('                   entry/sample/transformations/chi[()]: ', nxt_chi[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/chi not found')
    nxt_chi = None

try:
    nxt_kappa=fin['entry']['sample']['transformations']['kappa']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/kappa: ', nxt_kappa)
        print('                   entry/sample/transformations/kappa[()]: ', nxt_kappa[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/kappa not found')
    nxt_kappa = None

try:
    nxt_omega=fin['entry']['sample']['transformations']['omega']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/omega: ', nxt_omega)
        print('                   entry/sample/transformations/omega[()]: ', nxt_omega[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/omega not found')
    nxt_omega = None

try:
    nxt_phi=fin['entry']['sample']['transformations']['phi']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/phi: ', nxt_phi)
        print('                   entry/sample/transformations/phi[()]: ', nxt_phi[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/phi not found')
    nxt_phi = None

try:
    nxt_translation=fin['entry']['sample']['transformations']['translation']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/translation: ', nxt_translation)
        print('                   entry/sample/transformations/translation[()]: ', nxt_translation[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/translation not found')
    nxt_translation = None

# find {chi_end, kappa_end, omega_end, phi_end} in entry/sample/transformations/*
try:
    nxt_chi_end=fin['entry']['sample']['transformations']['chi_end']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/chi_end: ', nxt_chi_end)
        print('                   entry/sample/transformations/chi_end[()]: ', nxt_chi_end[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/chi_end not found')
    nxt_chi_end = None

try:
    nxt_kappa_end=fin['entry']['sample']['transformations']['kappa_end']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/kappa_end: ', nxt_kappa_end)
        print('                   entry/sample/transformations/kappa_end[()]: ', nxt_kappa_end[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/kappa_end not found')
    nxt_kappa_end = None

try:
    nxt_omega_end=fin['entry']['sample']['transformations']['omega_end']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/omega_end: ', nxt_omega_end)
        print('                   entry/sample/transformations/omega_end[()]: ', nxt_omega_end[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/omega_end not found')
    nxt_omega_end = None

try:
    nxt_phi_end=fin['entry']['sample']['transformations']['phi_end']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/phi_end: ', nxt_phi_end)
        print('                   entry/sample/transformations/phi_end[()]: ', nxt_phi_end[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/phi_end not found')
    nxt_phi_end = None


# find {chi_range_average, kappa_range_average, omega_range_average, phi_range_average} 
#     in entry/sample/transformations/*_end
try:
    nxt_chi_range_average=fin['entry']['sample']['transformations']['chi_range_average']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/chi_range_average: ', \
            nxt_chi_range_average)
        print('                   entry/sample/transformations/chi_range_average[()]: ', \
            nxt_chi_range_average[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/chi_range_average not found')
    nxt_chi_range_average = None

try:
    nxt_kappa_range_average=fin['entry']['sample']['transformations']['kappa_range_average']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/kappa_range_average: ', \
            nxt_kappa_range_average)
        print('                   entry/sample/transformations/kappa_range_average[()]: ', \
            nxt_kappa_range_average[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/kappa_range_average not found')
    nxt_kappa_range_average = None

try:
    nxt_omega_range_average=fin['entry']['sample']['transformations']['omega_range_average']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/omega_range_average: ', \
            nxt_omega_range_average)
        print('                   entry/sample/transformations/omega_range_average[()]: ', \
            nxt_omega_range_average[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/omega_range_average not found')
    nxt_omega_range_average = None

try:
    nxt_phi_range_average=fin['entry']['sample']['transformations']['phi_range_average']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/phi_range_average: ', \
            nxt_phi_range_average)
        print('                   entry/sample/transformations/phi_range_average[()]: ', \
            nxt_phi_range_average[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/phi_range_average not found')
    nxt_phi_range_average = None


# find {chi_range_total, kappa_range_total, omega_range_total, phi_range_total} 
#    in entry/sample/transformations/*_range_total
try:
    nxt_chi_range_total=fin['entry']['sample']['transformations']['chi_range_total']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/chi_range_total: ', \
            nxt_chi_range_total)
        print('                   entry/sample/transformations/chi_range_total[()]: ', \
            nxt_chi_range_total[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/chi_range_total not found')
    nxt_chi_range_total = None

try:
    nxt_kappa_range_total=fin['entry']['sample']['transformations']['kappa_range_total']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/kappa_range_total: ', \
            nxt_kappa_range_total)
        print('                   entry/sample/transformations/kappa_range_total[()]: ', \
            nxt_kappa_range_total[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/kappa_range_total not found')
    nxt_kappa_range_total = None

try:
    nxt_omega_range_total=fin['entry']['sample']['transformations']['omega_range_total']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/omega_range_total: ', \
            nxt_omega_range_total)
        print('                   entry/sample/transformations/omega_range_total[()]: ', \
            nxt_omega_range_total[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/omega_range_total not found')
    nxt_omega_range_total = None

try:
    nxt_phi_range_total=fin['entry']['sample']['transformations']['phi_range_total']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/transformations/phi_range_total: ', \
            nxt_phi_range_total)
        print('                   entry/sample/transformations/phi_range_total[()]: ', \
            nxt_phi_range_total[()])
except:
    print('l_bnl_compress.py: entry/sample/transformations/phi_range_total not found')
    nxt_phi_range_total = None

samp_gon = None
chi = None
chi_end = None
chi_range_average = None
chi_range_total = None
kappa = None
kappa_end = None
kappa_range_average = None
kappa_range_total = None
angles = None
angles_end = None
osc_width = None
osc_total = None
phi = None
phi_end = None
phi_range_average = None
phi_range_total = None
try:
    samp_gon = fin['entry']['sample']['goniometer']
    if args['verbose'] == True:
        print('l_bnl_compress.py: entry/sample/goniometer: ', samp_gon)
except:
    print('l_bnl_compress.py: entry/sample/goniometer not found')
    samp_gon = None
if samp_gon != None:
    try:
        chi=samp_gon['chi']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/chi: ', chi)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/chi not found')
        chi=None
    try:
        chi_end=samp_gon['chi_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/chi_end: ', chi_end)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/chi_end not found')
        chi_end=None
    try:
        chi_range_average = samp_gon['chi_range_average']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/chi_range_average: ', \
                chi_range_average)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/chi_range_average not found')
        chi_range_average=None
    try:
        chi_range_total = samp_gon['chi_range_total']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/chi_range_total: ', \
                chi_range_total)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/chi_range_total not found')
        chi_range_total=None
    try:
        kappa=samp_gon['kappa']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/kappa: ', kappa)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/kappa not found')
        kappa=None
    try:
        kappa_end=samp_gon['kappa_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/kappa_end: ', kappa_end)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/kappa_end not found')
        kappa_end=None
    try:
        kappa_range_average = samp_gon['kappa_range_average']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/kappa_range_average: ', \
                kappa_range_average)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/kappa_range_average not found')
        kappa_range_average=None
    try:
        kappa_range_total = samp_gon['kappa_range_total']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/kappa_range_total: ', \
                kappa_range_total)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/kappa_range_total not found')
        kappa_range_total=None
    try:
        angles=samp_gon['omega']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/omega: ', angles)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/omega not found')
        angles=None
    try:
        angles_end=samp_gon['omega_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/omega_end: ', angles_end)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/omega_end not found')
        angles_end=None
    try: 
        osc_width = samp_gon['omega_range_average']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/omega_range_average: ', osc_width)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer not found')
        osc_width = None
    try:
        osc_total = samp_gon['omega_range_total']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/omega_range_total: ', osc_total)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/omega_range_total not found')
        osc_total = None
    try:
        phi=samp_gon['phi']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/phi: ', phi)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/phi not found')
        phi=None
    try:
        phi_end=samp_gon['phi_end']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/phi_end: ', phi_end)
    except:
        print('l_bnl_compress.py: entry/sample/gonimeter/phi_end not found')
        phi_end=None
    try:
        phi_range_average = samp_gon['phi_range_average']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/phi_range_average: ', \
                phi_range_average)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/phi_range_average not found')
        phi_range_average=None
    try:
        phi_range_total = samp_gon['phi_range_total']
        if args['verbose'] == True:
            print('l_bnl_compress.py: entry/sample/goniometer/phi_range_total: ', \
                phi_range_total)
    except:
        print('l_bnl_compress.py: entry/sample/goniometer/phi_range_total not found')
        phi_range_total=None

try:
    datagroup=fin['entry']['data']
except:
    print('l_bnl_compress.py: entry/data not found')
    fin.close()
    sys.exit(-1)

block_start=0
try:
    data_start=datagroup['data_000000']
except:
    block_start=1
    try:
         data_start=datagroup['data_000001']
    except:
        print('l_bnl_compress.py: first data block not found')
        fin.close()
        sys.exit(-1)

block_shape=data_start.shape
if  len(block_shape) != 3:
    print('compress.py: dimension of /entry/data data block is not 3')
    fin.close()
    sys.exit

number_per_block=int(block_shape[0])
if args['verbose']==True:
    print('l_bnl_compress.py: first data block: ', data_start)
    print('                 dir(first data block): ', dir(data_start))
    print('                 number_per_block: ',number_per_block)

print (args['first_image'],args['last_image']+1,args['sum_range'])

if (args['sum_range'] == None) or (args['sum_range']  < 2):
    args['sum_range'] = 1
if (args['bin_range'] == None) or (args['bin_range'] < 2):
    args['bin_range'] = 1

new_nimage = 0
new_images = {}

print("args['first_image']",args['first_image'])
print("(args['last_image'])+1",(args['last_image'])+1)
print("args['sum_range']",args['sum_range'])

for image in range(args['first_image'],(args['last_image'])+1,args['sum_range']):
    lim_image=image+int(args['sum_range'])
    if lim_image > args['last_image']+1:
        lim_image = args['last_image']+1
    if args['verbose']==True:
        print('Adding images from ',image,' to ',lim_image)
    prev_out=None
    for cur_image in range(image,lim_image):
        if args['verbose']==True:
            print('image, (block,offset): ',cur_image,\
              conv_image_to_block_offset(cur_image,number_per_block))
        cur_source=conv_image_to_block_offset(cur_image,number_per_block)
        cur_source_img_block='data_'+str(cur_source[0]+block_start).zfill(6)
        cur_source_img_imgno=cur_source[1]
        cur_source=datagroup[cur_source_img_block][cur_source_img_imgno,:,:]
        if args['verbose']==True:
            print('image input shape ',cur_source.shape)
        print('cur_source_img_block: ', cur_source_img_block)
        print('cur_source_img_imgno: ', cur_source_img_imgno)
        if args['bin_range'] > 1:
            cur_source=bin_array(cur_source,int(args['bin_range']),satval)
        if cur_image > image:
            prev_out = np.clip(prev_out+cur_source,0,satval)
        else:
            prev_out = (np.asarray(cur_source,dtype='i2')).clip(0,satval)
    new_nimage = new_nimage+1
    new_images[new_nimage]=prev_out
    if args['verbose']==True:
        print('image output shape ',new_nimage,' ',new_images[new_nimage].shape)


if (args['data_block_size'] == None) or (args['data_block_size'] < 2):
    args['data_block_size'] = 1
out_number_per_block = args['data_block_size']
out_number_of_blocks = int(new_nimage+out_number_per_block-1)//out_number_per_block
out_max_image=new_nimage
if args['verbose'] == True:
    print('out_number_per_block: ', out_number_per_block)
    print('out_number_of_blocks: ', out_number_of_blocks)
fout={}
if args['out_squash'] != None:
    fout_squash={}

# create the master file
master=0
if args['out_master']==None or args['out_master']=='out_file':
    args['out_master']=args['out_file']+"_master"
if args['out_squash']=='out_file':
    args['out_squash']=args['out_file']+"_squash"
fout[master] = h5py.File(args['out_master']+".h5",'w')
fout[master].attrs.create('default',ntstr('entry'),dtype=ntstrdt('entry'))
fout[master].create_group('entry') 
fout[master]['entry'].attrs.create('NX_class',ntstr('NXentry'),dtype=ntstrdt('NXentry'))
fout[master]['entry'].attrs.create('default',ntstr('data'),dtype=ntstrdt('data'))
fout[master]['entry'].create_group('data') 
fout[master]['entry']['data'].attrs.create('NX_class',ntstr('NXdata'),dtype=ntstrdt('NXdata'))
fout[master]['entry']['data'].attrs.create('signal',ntstr('data_000001'),\
    dtype=ntstrdt('data_000001'))
if top_definition != None:
    fout[master]['entry'].create_dataset('definition',shape=top_definition.shape,\
        dtype=top_definition.dtype)
    fout[master]['entry']['definition'][()]=top_definition[()]
    if 'version' in top_definition.attrs.keys():
        fout[master]['entry']['definition'].attrs.create('version',\
        top_definition.attrs['version'])
fout[master]['entry'].create_group('instrument') 
fout[master]['entry']['instrument'].attrs.create('NX_class',ntstr('NXinstrument'),dtype=ntstrdt('NXinstrument'))
fout[master]['entry'].create_group('sample') 
fout[master]['entry']['sample'].attrs.create('NX_class',ntstr('NXsample'),dtype=ntstrdt('NXsample')) 
fout[master]['entry']['sample'].create_group('goniometer') 
fout[master]['entry']['sample']['goniometer'].attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
fout[master]['entry']['sample'].create_group('transformations') 
fout[master]['entry']['sample']['transformations'].attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))  
fout[master]['entry']['instrument'].attrs.create('NX_class',ntstr('NXinstrument'),dtype=ntstrdt('NXinstrument'))
fout[master]['entry']['instrument'].create_group('detector')
fout[master]['entry']['instrument']['detector'].attrs.create(\
    'NX_class',ntstr('NXdetector'),dtype=ntstrdt('NXdetector'))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'description',shape=description.shape,dtype=description.dtype)
fout[master]['entry']['instrument']['detector']['description'][()]=\
    description[()]
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'detector_number',shape=detector_number.shape,dtype=detector_number.dtype)
fout[master]['entry']['instrument']['detector']['detector_number'][()]=\
    detector_number[()]
if depends_on != None:
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'depends_on',shape=depends_on.shape,dtype=depends_on.dtype)
    fout[master]['entry']['instrument']['detector']['depends_on'][()]=\
        depends_on[()]
if bit_depth_image != None:
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'bit_depth_image',shape=bit_depth_image.shape,dtype='u4')
    fout[master]['entry']['instrument']['detector']['bit_depth_image'][()]=\
        np.uint32(16)
    fout[master]['entry']['instrument']['detector']['bit_depth_image'].attrs.create(\
        'units',ntstr('NX_UINT32'),dtype=ntstrdt('NX_UINT32'))
if bit_depth_readout != None:
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'bit_depth_readout',shape=bit_depth_readout.shape,dtype=bit_depth_readout.dtype)
    fout[master]['entry']['instrument']['detector']['bit_depth_readout'][()]=\
        bit_depth_readout[()]
    fout[master]['entry']['instrument']['detector']['bit_depth_readout'].attrs.create(\
        'units',ntstr('NX_UINT32'),dtype=ntstrdt('NX_UINT32'))
if countrate_correction_applied != None:
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'countrate_correction_applied',shape=countrate_correction_applied.shape,dtype=countrate_correction_applied.dtype)
    fout[master]['entry']['instrument']['detector']['countrate_correction_applied'][()]=\
        countrate_correction_applied[()]
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'sensor_thickness',shape=thickness.shape,dtype=thickness.dtype)
fout[master]['entry']['instrument']['detector']['sensor_thickness'][()]=\
    thickness[()]
fout[master]['entry']['instrument']['detector']['sensor_thickness'].attrs.create(\
    'units',ntstr(thickness.attrs['units']),dtype=ntstrdt(thickness.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'beam_center_x',shape=beamx.shape,dtype=beamx.dtype)
fout[master]['entry']['instrument']['detector']['beam_center_x'][()]=\
    beamx[()]/int(args['bin_range'])
fout[master]['entry']['instrument']['detector']['beam_center_x'].attrs.create(\
    'units',ntstr(beamx.attrs['units']),dtype=ntstrdt(beamx.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'beam_center_y',shape=beamy.shape,dtype=beamy.dtype)
fout[master]['entry']['instrument']['detector']['beam_center_y'][()]\
    =beamy[()]/int(args['bin_range'])
fout[master]['entry']['instrument']['detector']['beam_center_y'].attrs.create(\
   'units',ntstr(beamy.attrs['units']),dtype=ntstrdt(beamy.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'count_time',shape=count_time.shape,dtype=count_time.dtype)
fout[master]['entry']['instrument']['detector']['count_time'][()]=\
    count_time[()]*int(args['sum_range'])
fout[master]['entry']['instrument']['detector']['count_time'].attrs.create(\
    'units',ntstr(count_time.attrs['units']),dtype=ntstrdt(count_time.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'detector_distance',shape=distance.shape,dtype=distance.dtype)
fout[master]['entry']['instrument']['detector']['detector_distance'][()]=\
    distance[()]
fout[master]['entry']['instrument']['detector']['detector_distance'].attrs.create(\
    'units',ntstr(distance.attrs['units']),dtype=ntstrdt(distance.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'frame_time',shape=frame_time.shape,dtype=frame_time.dtype)
fout[master]['entry']['instrument']['detector']['frame_time'][()]=\
    frame_time[()]*int(args['sum_range'])
fout[master]['entry']['instrument']['detector']['frame_time'].attrs.create(\
    'units',ntstr(frame_time.attrs['units']),\
    dtype=ntstrdt(frame_time.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'x_pixel_size',shape=pixelsizex.shape,dtype=pixelsizex.dtype)
fout[master]['entry']['instrument']['detector']['x_pixel_size'][()]=\
    pixelsizex[()]*int(args['sum_range'])
fout[master]['entry']['instrument']['detector']['x_pixel_size'].attrs.create(\
    'units',ntstr(pixelsizex.attrs['units']),\
    dtype=ntstrdt(pixelsizex.attrs['units']))
fout[master]['entry']['instrument']['detector'].create_dataset(\
    'y_pixel_size',shape=pixelsizey.shape,dtype=pixelsizey.dtype)
fout[master]['entry']['instrument']['detector']['y_pixel_size'][()]=\
    pixelsizey[()]*int(args['sum_range'])
fout[master]['entry']['instrument']['detector']['y_pixel_size'].attrs.create(\
    'units',ntstr(pixelsizey.attrs['units']),\
    dtype=ntstrdt(pixelsizey.attrs['units']))
if pixel_mask!=None:
    new_pixel_mask=conv_pixel_mask(pixel_mask,int(args['bin_range']))
    fout[master]['entry']['instrument']['detector'].create_dataset(\
        'pixel_mask',shape=new_pixel_mask.shape,dtype='u4',\
        data=new_pixel_mask,\
        **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    del new_pixel_mask
fout[master]['entry']['instrument']['detector'].create_group(\
    'detectorSpecific')
fout[master]['entry']['instrument']['detector']['detectorSpecific'].attrs.create(\
    'NX_class',ntstr('NXcollection'))
new_shape=conv_image_shqpe((int(ypixels[()]),int(xpixels[()])),int(args['bin_range']))
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'auto_summation',data=1,dtype='i1')
print('compression: ',args['compression'])
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'compression',data=args['compression'])
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'nimages',shape=xnimages.shape,dtype=xnimages.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific']['nimages'][()]\
    =new_nimage
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'ntrigger',shape=xntrigger.shape,dtype=xntrigger.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific']['ntrigger'][()]=\
    new_nimage
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'x_pixels_in_detector',shape=xpixels.shape,dtype=xpixels.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific'][\
    'x_pixels_in_detector'][()]=new_shape[1]
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'y_pixels_in_detector',shape=ypixels.shape,dtype=ypixels.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific']['y_pixels_in_detector'][()]=\
    new_shape[0]
fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'software_version',shape=software_version.shape,dtype=software_version.dtype)
fout[master]['entry']['instrument']['detector']['detectorSpecific']['software_version'][()]=\
    software_version[()]
if satval_not_found:
    fout[master]['entry']['instrument']['detector']['detectorSpecific']['saturation_value'][()]=\
    satval
    fout[master]['entry']['instrument']['detector']['detectorSpecific']['countrate_correction_count_cutoff'][()]=\
    satval
if saturation_value != None:
    fout[master]['entry']['instrument']['detector'].create_dataset(\
    'saturation_value',shape=saturation_value.shape,dtype=saturation_value.dtype)
    fout[master]['entry']['instrument']['detector']['saturation_value'][()]=saturation_value[()]
if countrate_correction_count_cutoff != None:
    fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'countrate_correction_count_cutoff',shape=countrate_correction_count_cutoff.shape,\
    dtype=countrate_correction_count_cutoff.dtype)
    fout[master]['entry']['instrument']['detector']['detectorSpecific']['countrate_correction_count_cutoff'][()]=\
        countrate_correction_count_cutoff[()]
if dS_saturation_value != None:
    fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
    'saturation_value',shape=dS_saturation_value.shape,dtype=dS_saturation_value.dtype)
    fout[master]['entry']['instrument']['detector']['detectorSpecific']['saturation_value'][()]=\
    dSsaturation_value[()]

if det_gon != None:
    fout[master]['entry']['instrument']['detector'].create_group('goniometer')
    fout[master]['entry']['instrument']['detector']['goniometer'].attrs.create('NX_class',\
    ntstr('NXgoniometer'),dtype=ntstrdt('NXgoniometer'))
    if det_gon_two_theta != None:
        newshape = det_gon_two_theta.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['instrument']['detector']['goniometer'].create_dataset(\
        'two_theta',shape=newshape,dtype=det_gon_two_theta.dtype)
        fout[master]['entry']['instrument']['detector']['goniometer']['two_theta'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        det_gon_two_theta[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['goniometer']['two_theta'],\
        det_gon_two_theta)
    if det_gon_two_theta_end != None:
        newshape = det_gon_two_theta_end.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['instrument']['detector']['goniometer'].create_dataset(\
        'two_theta_end',shape=newshape,dtype=det_gon_two_theta_end.dtype)
        fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_end'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        det_gon_two_theta_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_end'],\
        det_gon_two_theta_end)
    if det_gon_two_theta_range_average != None:
        newshape = det_gon_two_theta_range_average.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['instrument']['detector']['goniometer'].create_dataset(\
        'two_theta_range_average',shape=newshape,dtype=det_gon_two_theta_range_average.dtype)
        fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_range_average'][()]=\
        det_gon_two_theta_range_average[()]*int(args['sum_range'])
        xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_range_average'],\
        det_gon_two_theta_range_average)
    if det_gon_two_theta_range_total != None:
        newshape = det_gon_two_theta_range_total.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['instrument']['detector']['goniometer'].create_dataset(\
        'two_theta_range_total',shape=newshape,dtype=det_gon_two_theta_range_total.dtype)
        fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_range_total'][()]=\
        det_gon_two_theta_range_total[()]
        xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['goniometer']['two_theta_range_total'],\
        det_gon_two_theta_range_total)

if det_nxt != None:
    fout[master]['entry']['instrument']['detector'].create_group('transformations')
    fout[master]['entry']['instrument']['detector']['transformations'].attrs.create('NX_class',\
    ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    if det_nxt_translation != None:
        newshape = det_nxt_translation.shape
        if newshape[0]==nimages:
            newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
        fout[master]['entry']['instrument']['detector']['transformations'].create_dataset(\
        'translation',shape=newshape,dtype=det_nxt_translation.dtype)
        fout[master]['entry']['instrument']['detector']['transformations']['translation'][()]=\
        det_nxt_translation[()]
        xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['transformations']['translation'],\
        det_nxt_translation)
    if det_nxt_two_theta != None:
        fout[master]['entry']['instrument']['detector']['transformations'].create_dataset(\
        'two_theta',shape=det_nxt_two_theta.shape,dtype=det_nxt_two_theta.dtype)
        fout[master]['entry']['instrument']['detector']['transformations']['two_theta'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        det_nxt_two_theta[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['transformations']['two_theta'],\
        det_nxt_two_theta)
    if det_nxt_two_theta_end != None:
        fout[master]['entry']['instrument']['detector']['transformations'].create_dataset(\
        'two_theta_end',shape=det_nxt_two_theta_end.shape,dtype=det_nxt_two_theta_end.dtype)
        fout[master]['entry']['instrument']['detector']['transformations']['two_theta_end'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        det_nxt_two_theta_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
        xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['transformations']['two_theta_end'],\
        det_nxt_two_theta_end)
    if det_nxt_two_theta_range_average != None:
        fout[master]['entry']['instrument']['detector']['transformations'].create_dataset(\
        'two_theta_range_average',shape=det_nxt_two_theta_range_average.shape,dtype=det_nxt_two_theta_range_average.dtype)
        fout[master]['entry']['instrument']['detector']['transformations']['two_theta_range_average'][()]=\
        det_nxt_two_theta_range_average[()]*int(args['sum_range'])
        xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['transformations']['two_theta_range_average'],\
        det_nxt_two_theta_range_average)
    if det_nxt_two_theta_range_total != None:
        fout[master]['entry']['instrument']['detector']['transformations'].create_dataset(\
        'two_theta_range_total',shape=det_nxt_two_theta_range_total.shape,dtype=det_nxt_two_theta_range_total.dtype)
        fout[master]['entry']['instrument']['detector']['transformations']['two_theta_range_total'][()]=\
        det_nxt_two_theta_range_total[()]
        xfer_axis_attrs(fout[master]['entry']['instrument']['detector']['transformations']['two_theta_range_total'],\
        det_nxt_two_theta_range_total)

if mod0_countrate_cutoff != None:
    if not ('detectorModule_000' in fout[master]['entry']['instrument']['detector']['detectorSpecific'].keys()):
        fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_group('detectorModule_000')
    fout[master]['entry']['instrument']['detector']['detectorSpecific']['detectorModule_000'].create_dataset(\
    'countrate_correction_count_cutoff',shape=mod0_countrate_cutoff.shape,dtype=mod0_countrate_cutoff.dtype)
    fout[master]['entry']['instrument']['detector']['detectorSpecific']['detectorModule_000']\
['countrate_correction_count_cutoff'][()]=mod0_countrate_cutoff[()]
if dS_pixel_mask!=None:
    new_pixel_mask=conv_pixel_mask(dS_pixel_mask,int(args['bin_range']))
    fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
        'pixel_mask',shape=new_pixel_mask.shape,dtype='u4',\
        data=new_pixel_mask,chunks=None,\
        **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    del new_pixel_mask
elif pixel_mask!=None:
    new_pixel_mask=conv_pixel_mask(pixel_mask,int(args['bin_range']))
    fout[master]['entry']['instrument']['detector']['detectorSpecific'].create_dataset(\
        'pixel_mask',shape=pixel_mask.shape,dtype='u4',\
        data=new_pixel_mask,chunks=None,\
        **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    del new_pixel_mask

if sample_wavelength!=None:
    if not ('beam' in fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('beam') 
        fout[master]['entry']['sample']['beam'].attrs.create('NX_class',ntstr('NXbeam'),dtype=ntstrdt('NXbeam'))
    if not ('incident_wavelength' in  fout[master]['entry']['sample']['beam'].keys()): 
        fout[master]['entry']['sample']['beam'].create_dataset(\
        'incident_wavelength',shape=sample_wavelength.shape,dtype=sample_wavelength.dtype)
    fout[master]['entry']['sample']['beam']['incident_wavelength'][()]=sample_wavelength[()]
    if 'units' in sample_wavelength.attrs.keys():
        fout[master]['entry']['sample']['beam']['incident_wavelength'].attrs.create('units',\
            sample_wavelength.attrs['units'])
if instrument_wavelength!=None:
    if not ('beam' in fout[master]['entry']['instrument'].keys()):
        fout[master]['entry']['instrument'].create_group('beam')
        fout[master]['entry']['instrument']['beam'].attrs.create('NX_class',ntstr('NXbeam'),dtype=ntstrdt('NXbeam'))
    if not ('wavelength' in fout[master]['entry']['instrument']['beam'].keys()):
        fout[master]['entry']['instrument']['beam'].create_dataset(\
        'wavelength',shape=instrument_wavelength.shape,dtype=instrument_wavelength.dtype)
    fout[master]['entry']['instrument']['beam']['wavelength'][()]=instrument_wavelength[()]
    if 'units' in instrument_wavelength.attrs.keys():
        fout[master]['entry']['instrument']['beam']['wavelength'].attrs.create('units',\
            instrument_wavelength.attrs['units'])
if monochromater_wavelength!=None:
    if not ('monochromater' in fout[master]['entry']['instrument'].keys()):
        fout[master]['entry']['instrument'].create_group('monochromater')
        fout[master]['entry']['instrument']['monochromater'].attrs.create('NX_class',ntstr('NXmonochromater'),dtype=ntstrdt('NXmonochromater'))
    if not ('wavelength' in fout[master]['entry']['instrument']['monochromater'].keys()):
        fout[master]['entry']['instrument']['monochromater'].create_dataset(\
        'wavelength',shape=monochromater_wavelength.shape,dtype=monochromater_wavelength.dtype)
    fout[master]['entry']['instrument']['monochromater']['wavelength'][()]=monochromater_wavelength[()]
    if 'units' in monochromater_wavelength.attrs.keys():
        fout[master]['entry']['instrument']['monochromater']['wavelength'].attrs.create('units',\
            monochromater_wavelength.attrs['units'])
if beam_incident_wavelength!=None:
    if not ('beam' in fout[master]['entry']['instrument'].keys()):
        fout[master]['entry']['instrument'].create_group('beam')
        fout[master]['entry']['instrument']['beam'].attrs.create('NX_class',ntstr('NXbeam'),dtype=ntstrdt('NXbeam'))
    if not ('incident_wavelength' in fout[master]['entry']['instrument']['beam'].keys()):
        fout[master]['entry']['instrument']['beam'].create_dataset(\
        'incident_wavelength',shape=beam_incident_wavelength.shape,dtype=beam_incident_wavelength.dtype)
    fout[master]['entry']['instrument']['beam']['incident_wavelength'][()]=beam_incident_wavelength[()]
    if 'units' in beam_incident_wavelength.attrs.keys():
        fout[master]['entry']['instrument']['beam']['incident_wavelength'].attrs.create('units',\
            beam_incident_wavelength.attrs['units'])


if chi!=None:
    newshape = chi.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'chi',shape=newshape,dtype=chi.dtype) 
    fout[master]['entry']['sample']['goniometer']['chi'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    chi[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['goniometer']['chi'].attrs.create(\
        'units',ntstr(chi.attrs['units']),\
        dtype=ntstrdt(chi.attrs['units']))
if chi_end!=None:
    newshape = chi_end.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'chi_end',shape=newshape,dtype=chi_end.dtype)
    fout[master]['entry']['sample']['goniometer']['chi_end'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    chi_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['goniometer']['chi_end'].attrs.create(\
        'units',ntstr(chi_end.attrs['units']),\
        dtype=ntstrdt(chi_end.attrs['units']))
if chi_range_average != None:
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'chi_range_average',shape=chi_range_average.shape,dtype=chi_range_average.dtype)
    fout[master]['entry']['sample']['goniometer']['chi_range_average'][()]=\
    chi_range_average[()]*int(args['sum_range'])
    fout[master]['entry']['sample']['goniometer']['chi_range_average'].attrs.create(\
        'units',ntstr(chi_range_average.attrs['units']),\
        dtype=ntstrdt(chi_range_average.attrs['units']))
if chi_range_total != None:
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'chi_range_total',shape=chi_range_average.shape,dtype=chi_range_total.dtype)
    fout[master]['entry']['sample']['goniometer']['chi_range_total'][()]=\
    chi_range_average[()]
    fout[master]['entry']['sample']['goniometer']['chi_range_total'].attrs.create(\
        'units',ntstr(chi_range_total.attrs['units']),\
        dtype=ntstrdt(chi_range_total.attrs['units']))
if kappa!=None:
    newshape = kappa.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'kappa',shape=newshape,dtype=kappa.dtype) 
    fout[master]['entry']['sample']['goniometer']['kappa'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    kappa[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['goniometer']['kappa'].attrs.create(\
        'units',ntstr(kappa.attrs['units']),\
        dtype=ntstrdt(kappa.attrs['units']))
if kappa_end!=None:
    newshape = kappa_end.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'kappa_end',shape=newshape,dtype=kappa_end.dtype)
    fout[master]['entry']['sample']['goniometer']['kappa_end'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    kappa_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['goniometer']['kappa_end'].attrs.create(\
        'units',ntstr(kappa_end.attrs['units']),\
        dtype=ntstrdt(kappa_end.attrs['units']))
if kappa_range_average != None:
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'kappa_range_average',shape=kappa_range_average.shape,dtype=kappa_range_average.dtype)
    fout[master]['entry']['sample']['goniometer']['kappa_range_average'][()]=\
    kappa_range_average[()]*int(args['sum_range'])
    fout[master]['entry']['sample']['goniometer']['kappa_range_average'].attrs.create(\
        'units',ntstr(kappa_range_average.attrs['units']),\
        dtype=ntstrdt(kappa_range_average.attrs['units']))
if kappa_range_total != None:
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'kappa_range_total',shape=kappa_range_total.shape,dtype=kappa_range_total.dtype)
    fout[master]['entry']['sample']['goniometer']['kappa_range_total'][()]=\
    kappa_range_total[()]
    fout[master]['entry']['sample']['goniometer']['kappa_range_total'].attrs.create(\
        'units',ntstr(kappa_range_total.attrs['units']),\
        dtype=ntstrdt(kappa_range_total.attrs['units']))
if angles != None:
    newshape = angles.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    print('angles.shape: ',angles.shape)
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'omega',shape=newshape,dtype=angles.dtype) 
    fout[master]['entry']['sample']['goniometer']['omega'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    angles[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['goniometer']['omega'].attrs.create(\
       'units',ntstr(angles.attrs['units']),\
       dtype=ntstrdt(angles.attrs['units']))
if angles_end != None:
    newshape = angles_end.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    print('angles_end.shape: ',angles_end.shape)
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'omega_end',shape=newshape,dtype=angles_end.dtype) 
    fout[master]['entry']['sample']['goniometer']['omega_end'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    angles_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['goniometer']['omega_end'].attrs.create(\
        'units',ntstr(angles_end.attrs['units']),\
        dtype=ntstrdt(angles_end.attrs['units']))
if osc_width != None:
    newshape = osc_width.shape
    print('osc_width.shape: ',osc_width.shape)
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'omega_range_average',shape=newshape,dtype=osc_width.dtype)
    fout[master]['entry']['sample']['goniometer']['omega_range_average'][()]=\
        osc_width[()]*int(args['sum_range'])
    fout[master]['entry']['sample']['goniometer']['omega_range_average'].attrs.create(\
        'units',ntstr(osc_width.attrs['units']),\
        dtype=ntstrdt(osc_width.attrs['units']))
if osc_total != None:
    newshape = osc_total.shape
    print('osc_total.shape: ',osc_total.shape)
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
        'omega_range_total',shape=newshape,dtype=osc_total.dtype)
    fout[master]['entry']['sample']['goniometer']['omega_range_total'][()]=\
        osc_total[()]
    fout[master]['entry']['sample']['goniometer']['omega_range_total'].attrs.create(\
        'units',ntstr(osc_total.attrs['units']),\
        dtype=ntstrdt(osc_total.attrs['units']))
if phi!=None:
    newshape = phi.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'phi',shape=newshape,dtype=phi.dtype) 
    fout[master]['entry']['sample']['goniometer']['phi'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    phi[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['goniometer']['phi'].attrs.create(\
        'units',ntstr(phi.attrs['units']),\
        dtype=ntstrdt(phi.attrs['units']))
if phi_end!=None:
    newshape = phi_end.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'phi_end',shape=newshape,dtype=phi_end.dtype)
    fout[master]['entry']['sample']['goniometer']['phi_end'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    phi_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['goniometer']['phi_end'].attrs.create(\
        'units',ntstr(phi_end.attrs['units']),\
        dtype=ntstrdt(phi_end.attrs['units']))
if phi_range_average != None:
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'phi_range_average',shape=phi_range_average.shape,dtype=phi_range_average.dtype)
    fout[master]['entry']['sample']['goniometer']['phi_range_average'][()]=\
    phi_range_average[()]*int(args['sum_range'])
    fout[master]['entry']['sample']['goniometer']['phi_range_average'].attrs.create(\
        'units',ntstr(phi_range_average.attrs['units']),\
        dtype=ntstrdt(phi_range_average.attrs['units']))
if phi_range_total != None:
    fout[master]['entry']['sample']['goniometer'].create_dataset(\
    'phi_range_total',shape=phi_range_total.shape,dtype=phi_range_total.dtype)
    fout[master]['entry']['sample']['goniometer']['phi_range_total'][()]=\
    phi_range_total[()]
    fout[master]['entry']['sample']['goniometer']['phi_range_total'].attrs.create(\
        'units',ntstr(phi_range_total.attrs['units']),\
        dtype=ntstrdt(phi_range_total.attrs['units']))
if nxt_chi!=None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    newshape = nxt_chi.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'chi',shape=newshape,dtype=nxt_chi.dtype)
    fout[master]['entry']['sample']['transformations']['chi'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        nxt_chi[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['transformations']['chi'].attrs.create(\
        'units',ntstr(nxt_chi.attrs['units']),\
        dtype=ntstrdt(nxt_chi.attrs['units']))
if nxt_chi_end!=None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    newshape = nxt_chi_end.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'chi_end',shape=newshape,dtype=nxt_chi_end.dtype)
    fout[master]['entry']['sample']['transformations']['chi_end'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        nxt_chi_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['transformations']['chi_end'].attrs.create(\
        'units',ntstr(nxt_chi_end.attrs['units']),\
        dtype=ntstrdt(nxt_chi_end.attrs['units']))
if nxt_chi_range_average != None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'chi_range_average',shape=nxt_chi_range_average.shape,dtype=nxt_chi_range_average.dtype)
    fout[master]['entry']['sample']['transformations']['chi_range_average'][()]=\
    nxt_chi_range_average[()]*int(args['sum_range'])
    fout[master]['entry']['sample']['transformations']['chi_range_average'].attrs.create(\
        'units',ntstr(nxt_chi_range_average.attrs['units']),\
        dtype=ntstrdt(nxt_chi_range_average.attrs['units']))
if nxt_chi_range_total != None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'chi_range_total',shape=nxt_chi_range_total.shape,dtype=nxt_chi_range_total.dtype)
    fout[master]['entry']['sample']['transformations']['chi_range_total'][()]=\
    nxt_chi_range_total[()]
    fout[master]['entry']['sample']['transformations']['chi_range_total'].attrs.create(\
        'units',ntstr(nxt_chi_range_total.attrs['units']),\
        dtype=ntstrdt(nxt_chi_range_total.attrs['units']))
if nxt_omega!=None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    newshape = nxt_omega.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'omega',shape=newshape,dtype=nxt_omega.dtype)
    fout[master]['entry']['sample']['transformations']['omega'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        nxt_omega[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    if 'depends_on' in nxt_omega.attrs.keys():
        fout[master]['entry']['sample']['transformations']['omega']\
            .attrs.create('depends_on',nxt_omega.attrs['depends_on'])
    if 'offset' in nxt_omega.attrs.keys():
        fout[master]['entry']['sample']['transformations']['omega']\
            .attrs.create('offset',nxt_omega.attrs['offset'])
    if 'transformation_type' in nxt_omega.attrs.keys():
        fout[master]['entry']['sample']['transformations']['omega']\
            .attrs.create('transformation_type',nxt_omega.attrs['transformation_type'])
    if 'units' in nxt_omega.attrs.keys():
        fout[master]['entry']['sample']['transformations']['omega']\
            .attrs.create('units',nxt_omega.attrs['units'])
    if 'vector' in nxt_omega.attrs.keys():
        fout[master]['entry']['sample']['transformations']['omega']\
            .attrs.create('vector',nxt_omega.attrs['vector'])

if nxt_omega_end!=None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    newshape = nxt_omega_end.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'omega_end',shape=newshape,dtype=nxt_omega_end.dtype)
    fout[master]['entry']['sample']['transformations']['omega_end'][0:(int(args['last_image'])-\
        int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
        nxt_omega_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['transformations']['omega_end'].attrs.create(\
        'units',ntstr(nxt_omega_end.attrs['units']),\
        dtype=ntstrdt(nxt_omega_end.attrs['units']))
if nxt_omega_range_average != None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'omega_range_average',shape=nxt_omega_range_average.shape,dtype=nxt_omega_range_average.dtype)
    fout[master]['entry']['sample']['transformations']['omega_range_average'][()]=\
    nxt_omega_range_average[()]*int(args['sum_range'])
    fout[master]['entry']['sample']['transformations']['omega_range_average'].attrs.create(\
        'units',ntstr(nxt_omega_range_average.attrs['units']),\
        dtype=ntstrdt(nxt_omega_range_average.attrs['units']))
if nxt_omega_range_total != None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'omega_range_total',shape=nxt_omega_range_total.shape,dtype=nxt_omega_range_total.dtype)
    fout[master]['entry']['sample']['transformations']['omega_range_total'][()]=\
    nxt_omega_range_total[()]
    fout[master]['entry']['sample']['transformations']['omega_range_total'].attrs.create(\
        'units',ntstr(nxt_omega_range_total.attrs['units']),\
        dtype=ntstrdt(nxt_omega_range_total.attrs['units']))
if nxt_phi_range_average != None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'phi_range_average',shape=nxt_phi_range_average.shape,dtype=nxt_phi_range_average.dtype)
    fout[master]['entry']['sample']['transformations']['phi_range_average'][()]=\
    nxt_phi_range_average[()]*int(args['sum_range'])
    fout[master]['entry']['sample']['transformations']['phi_range_average'].attrs.create(\
        'units',ntstr(nxt_phi_range_average.attrs['units']),\
        dtype=ntstrdt(nxt_phi_range_average.attrs['units']))
if nxt_phi!=None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    newshape = nxt_phi.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'phi',shape=newshape,dtype=nxt_phi.dtype)
    fout[master]['entry']['sample']['transformations']['phi'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    nxt_phi[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['transformations']['phi'].attrs.create(\
        'units',ntstr(nxt_phi.attrs['units']),\
        dtype=ntstrdt(nxt_phi.attrs['units']))
if nxt_phi_end!=None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    newshape = nxt_phi_end.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'phi_end',shape=newshape,dtype=nxt_phi_end.dtype)
    fout[master]['entry']['sample']['transformations']['phi_end'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    nxt_phi_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['transformations']['phi_end'].attrs.create(\
        'units',ntstr(nxt_phi_end.attrs['units']),\
        dtype=ntstrdt(nxt_phi_end.attrs['units']))
if nxt_kappa!=None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    newshape = nxt_kappa.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'kappa',shape=nxt_kappa.shape,dtype=nxt_kappa.dtype)
    fout[master]['entry']['sample']['transformations']['kappa'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    nxt_kappa[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['transformations']['kappa'].attrs.create(\
        'units',ntstr(nxt_kappa.attrs['units']),\
        dtype=ntstrdt(nxt_kappa.attrs['units']))
if nxt_kappa_end!=None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    newshape = nxt_kappa_end.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'kappa_end',shape=newshape,dtype=nxt_kappa_end.dtype)
    fout[master]['entry']['sample']['transformations']['kappa_end'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    nxt_kappa_end[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    fout[master]['entry']['sample']['transformations']['kappa_end'].attrs.create(\
        'units',ntstr(nxt_kappa_end.attrs['units']),\
        dtype=ntstrdt(nxt_kappa_end.attrs['units']))
if nxt_kappa_range_total != None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'kappa_range_total',shape=nxt_kappa_range_total.shape,dtype=nxt_kappa_range_total.dtype)
    fout[master]['entry']['sample']['transformations']['kappa_range_total'][()]=\
    nxt_kappa_range_total[()]
    fout[master]['entry']['sample']['transformations']['kappa_range_total'].attrs.create(\
        'units',ntstr(nxt_kappa_range_total.attrs['units']),\
        dtype=ntstrdt(nxt_kappa_range_total.attrs['units']))
if nxt_translation!=None:
    if not ('transformations' in  fout[master]['entry']['sample'].keys()):
        fout[master]['entry']['sample'].create_group('transformations')
        fout[master]['entry']['sample']['transformations'].\
            attrs.create('NX_class',ntstr('NXtransformations'),dtype=ntstrdt('NXtransformations'))
    newshape = nxt_translation.shape
    if newshape[0]==nimages:
        newshape=((newshape[0]+int(args['sum_range'])-1)/int(args['sum_range']),)+newshape[1:]
    fout[master]['entry']['sample']['transformations'].create_dataset(\
    'translation',shape=newshape,dtype=nxt_translation.dtype)
    fout[master]['entry']['sample']['transformations']['translation'][0:(int(args['last_image'])-\
    int(args['first_image'])+int(args['sum_range']))//int(args['sum_range'])]=\
    nxt_translation[int(args['first_image'])-1:int(args['last_image']):int(args['sum_range'])]
    if 'depends_on' in nxt_translation.attrs.keys():
        fout[master]['entry']['sample']['transformations']['translation']\
            .attrs.create('depends_on',nxt_translation.attrs['depends_on'])
    if 'offset' in nxt_translation.attrs.keys():
        fout[master]['entry']['sample']['transformations']['translation']\
            .attrs.create('offset',nxt_translation.attrs['offset'])
    if 'transformation_type' in nxt_translation.attrs.keys():
        fout[master]['entry']['sample']['transformations']['translation']\
            .attrs.create('transformation_type',nxt_translation.attrs['transformation_type'])
    if 'units' in nxt_translation.attrs.keys():
        fout[master]['entry']['sample']['transformations']['translation']\
            .attrs.create('units',nxt_translation.attrs['units'])
    if 'vector' in nxt_translation.attrs.keys():
        fout[master]['entry']['sample']['transformations']['translation']\
            .attrs.create('vector',nxt_translation.attrs['vector'])
for nout_block in range(1,out_number_of_blocks+1):
    nout_image=1+(nout_block-1)*out_number_per_block
    image_nr_low=nout_image
    lim_nout_image = nout_image+out_number_per_block
    if lim_nout_image > out_max_image+1:
        lim_nout_image = out_max_image+1
    image_nr_high = lim_nout_image-1
    nout_data_shape = new_images[nout_image].shape
    fout[nout_block] = h5py.File(args['out_file']+"_"+str(nout_block).zfill(6)+".h5",'w')
    fout[nout_block].create_group('entry')
    fout[nout_block]['entry'].attrs.create('NX_class',ntstr('NXentry'),dtype=ntstrdt('NXentry'))
    fout[nout_block]['entry'].create_group('data')
    fout[nout_block]['entry']['data'].attrs.create('NX_class',ntstr('NXdata'),dtype=ntstrdt('NXdata'))
    if args['out_squash'] != None:
        fout_squash[nout_block] = h5py.File(args['out_squash']+"_"+str(nout_block).zfill(6)+".h5",'w')
        fout_squash[nout_block].create_group('entry')
        fout_squash[nout_block]['entry'].attrs.create('NX_class',ntstr('NXentry'),dtype=ntstrdt('NXentry'))
        fout_squash[nout_block]['entry'].create_group('data')
        fout_squash[nout_block]['entry']['data'].attrs.create('NX_class',ntstr('NXdata'),dtype=ntstrdt('NXdata'))
    mydata_type='u2'
    if args['unit']==4:
        mydata_type='u4'
    if args['uint']==0
        mydata_type='i4'
    if args['compression']==None:
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]))
    elif args['compression']=='bshuf' or args['compression']=='BSHUF':
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='none'))
    elif args['compression']=='bslz4' or args['compression']=='BSLZ4':
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    elif args['compression']=='bszstd' or args['compression']=='BSZSTD':
        if args['compression_level'] == None:
            clevel=3
        elif int(args['compression_level']) > 22:
            clevel=22
        elif int(args['compression_level']) < -2:
            clevel=-2
        else:
            clevel=int(args['compression_level'])
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='zstd',clevel=clevel))
    elif args['compression']=='zstd' or args['compression']=='ZSTD':
        if args['compression_level'] == None:
            clevel=3
        elif int(args['compression_level']) > 22:
            clevel=22
        elif int(args['compression_level']) < -2:
            clevel=-2
        else:
            clevel=int(args['compression_level'])
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Blosc(cname='zstd',clevel=clevel,shuffle=hdf5plugin.Blosc.NOSHUFFLE))
    else:
        print('l_bnl_compress.py: unrecognized compression, reverting to bslz4')
        fout[nout_block]['entry']['data'].create_dataset('data',
            shape=((lim_nout_image-nout_image),nout_data_shape[0],nout_data_shape[1]),
            maxshape=(None,nout_data_shape[0],nout_data_shape[1]),
            dtype=mydata_type,chunks=(1,nout_data_shape[0],nout_data_shape[1]),
            **hdf5plugin.Bitshuffle(nelems=0,cname='lz4'))
    fout[nout_block]['entry']['data']['data'].attrs.create('image_nr_low',dtype=np.uint64,
        data=np.uint64(image_nr_low))
    fout[nout_block]['entry']['data']['data'].attrs.create('image_nr_high',dtype=np.uint64,
        data=np.uint64(image_nr_high))
    if args['out_squash'] != None:
        fout_squash[nout_block]['entry']['data'].attrs.create('image_nr_low',dtype=np.uint64,
            data=np.uint64(image_nr_low))
        fout[nout_block]['entry']['data']['data'].attrs.create('image_nr_high',dtype=np.uint64,
            data=np.uint64(image_nr_high))
    print("fout[nout_block]['entry']['data']['data']: ",fout[nout_block]['entry']['data']['data'])
    hcomp_tot_compimgsize=0
    nhcomp_img = 0
    j2k_tot_compimgsize=0
    nj2k_img=0
    for out_image in range(nout_image,lim_nout_image):
        if args['hcomp_scale']==None and args['j2k_target_compression_ratio']==None: 
            fout[nout_block]['entry']['data']['data'][out_image-nout_image,0:nout_data_shape[0],0:nout_data_shape[1]] \
              =np.clip(new_images[out_image][0:nout_data_shape[0],0:nout_data_shape[1]],0,satval)
        elif args['hcomp_scale']!=None:
            myscale=args['hcomp_scale']
            if myscale < 1 :
                myscale=16
            img16=np.asarray(new_images[out_image][0:nout_data_shape[0],0:nout_data_shape[1]],dtype='i2')
            img16=(np.clip(img16,0,satval)).astype('i2')
            fits_bytes, original_shape = compress_HCarray(img16,satval,myscale)
            if args['out_squash'] != None:
                fout_squash[nout_block]['entry']['data'].create_dataset('data_'+str(out_image).zfill(6),data=repr(fits_bytes))
            hdecomp_data=decompress_HCarray(fits_bytes,original_shape,scale=myscale)
            decompressed_data= (np.maximum(hdecomp_data,0).astype(np.uint16)).reshape((nout_data_shape[0],nout_data_shape[1]))
            decompressed_data = np.clip(decompressed_data,0,satval)
            hcomp_tot_compimgsize = hcomp_tot_compimgsize+sys.getsizeof(fits_bytes)
            nhcomp_img = nhcomp_img+1
            if args['verbose'] == True:
                print('l_bnl_compress.py: Hcompress sys.getsizeof(): ', sys.getsizeof(fits_bytes))
                print('                   decompressed_data sys.getsizeof: ', sys.getsizeof(decompressed_data))
            fout[nout_block]['entry']['data']['data'][out_image-nout_image, \
                0:nout_data_shape[0],0:nout_data_shape[1]] \
                = np.asarray(decompressed_data,dtype='u2')
            del decompressed_data
            del hdecomp_data
            del fits_bytes
            del img16
        else:
            mycrat=int(args['j2k_target_compression_ratio'])
            if mycrat < 1:
                mycrat=125
            img16=new_images[out_image][0:nout_data_shape[0],0:nout_data_shape[1]].astype('u2')
            outtemp=args['out_file']+"_"+str(out_image).zfill(6)+".j2k"
            print("outtemp: ",outtemp)
            j2k=glymur.Jp2k(outtemp, data=img16, cratios=[mycrat])
            print ('j2k.dtype', j2k.dtype)
            print ('j2k.shape', j2k.shape)
            jdecomped = glymur.Jp2k(outtemp)
            jdecomped = np.maximum(0,jdecomped[:])
            arr_final = np.array(jdecomped, dtype='u2')
            file_size = os.path.getsize(outtemp)
            if args['out_squash'] != None:
                fout_squash[nout_block]['entry']['data']\
                    .create_dataset('data_'+str(out_image).zfill(6),data=jdecomped)
                if args['verbose'] == True:
                    print (fout_squash[nout_block]['entry']['data']['data_'+str(out_image).zfill(6)])
                fout_squash[nout_block]['entry']['data']['data_'+str(out_image).zfill(6)]\
                .attrs.create('compression',ntstr('j2k'),dtype=ntstrdt('j2k'))
                fout_squash[nout_block]['entry']['data']['data_'+str(out_image).zfill(6)]\
                .attrs.create('compression_level',mycrat,dtype=ntstrdt('i2'))
            j2k_tot_compimgsize = j2k_tot_compimgsize + file_size
            nj2k_img = nj2k_img+1
            if args['verbose'] == True:
                print('l_bnl_compress.py: JPEG-2000 outtemp file_size: ', file_size )
            fout[nout_block]['entry']['data']['data'][out_image-nout_image, \
                0:nout_data_shape[0],0:nout_data_shape[1]] \
                = np.clip(arr_final,0,satval)
            del arr_final
            del jdecomped
            del j2k
            os.remove(outtemp)
    fout[nout_block].close()
    if nhcomp_img > 0:
        print('l_bnl_compress.py: hcomp avg compressed image size: ', int(.5+hcomp_tot_compimgsize/nhcomp_img))
    if nj2k_img > 0:
        print('l_bnl_compress.py: j2k avg compressed imgage size: ', int(.5+j2k_tot_compimgsize/nj2k_img))
    fout[master]['entry']['data']["data_"+str(nout_block).zfill(6)] \
        = h5py.ExternalLink(os.path.basename(args['out_file'])+"_"+str(nout_block).zfill(6)+".h5", "/entry/data/data")
    if args['out_squash'] != None:
        fout[master]['entry']['data']["squash_"+str(nout_block).zfill(6)] \
        = h5py.ExternalLink(args['out_squash']+"_"+str(nout_block).zfill(6)+".h5", "/entry/data")
fout[master].close()
