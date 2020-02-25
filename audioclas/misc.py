import os
import mimetypes
import tarfile
import zipfile


def open_compressed(byte_stream, file_format, output_folder):
    """
    Extract and save a stream of bytes of a compressed file from memory.

    Parameters
    ----------
    byte_stream : BinaryIO
    file_format : str
        Compatible file formats: tarballs, zip files
    output_folder : str
        Folder to extract the stream

    Returns
    -------
    Folder name of the extracted files.
    """
    print('Decompressing the file ...')
    tar_extensions = ['tar', 'bz2', 'tb2', 'tbz', 'tbz2', 'gz', 'tgz', 'lz', 'lzma', 'tlz', 'xz', 'txz', 'Z', 'tZ']
    if file_format in tar_extensions:
        tar = tarfile.open(mode="r:{}".format(file_format), fileobj=byte_stream)
        tar.extractall(output_folder)
        folder_name = tar.getnames()[0]
        return os.path.join(output_folder, folder_name)

    elif file_format == 'zip':
        zf = zipfile.ZipFile(byte_stream)
        zf.extractall(output_folder)
        # folder_name = zf.namelist()[0].split('/')[0]
        # return os.path.join(output_folder, folder_name)

    else:
        raise ValueError('Invalid file format for the compressed byte_stream')


def is_audio(path):
    mimestart = mimetypes.guess_type(path)[0]
    if mimestart is not None and mimestart.split('/')[0] == 'audio':
        return True
    else:
        return False


def find_audiofiles(folder_path):
    """Find all audio files inside folder"""
    fpaths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            # Check if the file is an audio
            mimestart = mimetypes.guess_type(path)[0]
            if mimestart is not None and mimestart.split('/')[0] == 'audio':
                fpaths.append(path)
    return fpaths
