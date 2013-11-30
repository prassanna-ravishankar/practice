from distutils.core import setup,Extension

eigen_dir = ""
python_dir = ""
boost_dir = ""

python_lib_dir = ""
boost_lib_dir = ""

setup(name='pyeigen',
        ext_modules=[Extension('pyeigen',
                sources=["wrapper_example.cpp"],
                libraries=['boost_python', 'python2.6'],
                include_dirs=[eigen_dir, python_dir, boost_dir],
                library_dirs=[python_lib_dir, boost_lib_dir]               
        )],
        install_requires=['distribute']
)
