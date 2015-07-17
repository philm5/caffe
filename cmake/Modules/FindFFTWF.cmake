# - Find FFTWF
# Find the native FFTWF includes and library
#
#  FFTW_INCLUDES    - where to find fftw3.h
#  FFTWF_LIBRARIES   - List of libraries when using FFTW.
#  FFTW_FOUND       - True if FFTW found.

if (FFTW_INCLUDES)
  # Already in cache, be silent
  set (FFTW_FIND_QUIETLY TRUE)
endif (FFTW_INCLUDES)

find_path (FFTW_INCLUDES fftw3.h)

find_library (FFTWF_LIBRARIES NAMES fftw3f)

# handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (FFTWF DEFAULT_MSG FFTWF_LIBRARIES FFTW_INCLUDES)

mark_as_advanced (FFTWF_LIBRARIES FFTW_INCLUDES)