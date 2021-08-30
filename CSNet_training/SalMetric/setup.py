from distutils.core import Extension, setup

from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
    name="salmetric",
    version="0.1",
    description=
    "Measure F-measure and MAE scores for salient object detection.",
    author="Qibin (Andrew) Hou",
    author_email="andrewhoux@gmail.com",
    ext_modules=cythonize([
        Extension(
            "build.salmetric",
            sources=["./python/salmetric.pyx", "./src/sal_metric.cpp"],
            include_dirs=[
                './include', '/usr/local/include/',
                '/usr/local/include/opencv2/'
            ],
            libraries=[
                'opencv_core', 'opencv_highgui', 'opencv_imgproc',
                'opencv_imgcodecs'
            ],
            library_dirs=['/usr/local/lib/'],
            language="c++",
        ),
    ]),
    cmdclass={'build_ext': build_ext},
)
