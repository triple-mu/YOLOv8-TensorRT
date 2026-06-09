# Locate a TensorRT installation and expose the imported target TensorRT::TensorRT.
#
# Defines:
#   TensorRT_INCLUDE_DIRS / TensorRT_LIBRARIES / TensorRT_FOUND
#   TensorRT_VERSION_STRING / _MAJOR / _MINOR / _PATCH
#
# Hint: set TensorRT_ROOT to point at a specific install (e.g. /data/TensorRT-10.16.1.11).
# In addition to /usr, any /data/TensorRT-* directory is searched automatically.

set(_TensorRT_SEARCHES)

if(TensorRT_ROOT)
    set(_TensorRT_SEARCH_ROOT PATHS ${TensorRT_ROOT} NO_DEFAULT_PATH)
    list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_ROOT)
endif()

# Custom installs shipped under /data (tar packages), newest first.
file(GLOB _TensorRT_DATA_ROOTS /data/TensorRT-*)
list(SORT _TensorRT_DATA_ROOTS ORDER DESCENDING)
set(_TensorRT_SEARCH_NORMAL PATHS ${_TensorRT_DATA_ROOTS} "/usr")
list(APPEND _TensorRT_SEARCHES _TensorRT_SEARCH_NORMAL)

foreach(search ${_TensorRT_SEARCHES})
    find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h ${${search}} PATH_SUFFIXES include)
endforeach()

if(NOT TensorRT_LIBRARY)
    foreach(search ${_TensorRT_SEARCHES})
        find_library(TensorRT_LIBRARY NAMES nvinfer ${${search}} PATH_SUFFIXES lib)
        if(NOT TensorRT_LIB_DIR)
            get_filename_component(TensorRT_LIB_DIR ${TensorRT_LIBRARY} DIRECTORY)
        endif()
    endforeach()
endif()

if(NOT TensorRT_nvinfer_plugin_LIBRARY)
    foreach(search ${_TensorRT_SEARCHES})
        find_library(TensorRT_nvinfer_plugin_LIBRARY NAMES nvinfer_plugin ${${search}} PATH_SUFFIXES lib)
    endforeach()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR)

# Resolve one version component. Handles both the public headers (numeric literal,
# e.g. `#define NV_TENSORRT_MAJOR 10`) and the enterprise headers, which define it
# indirectly (`#define NV_TENSORRT_MAJOR TRT_MAJOR_ENTERPRISE` with the number on a
# separate `#define TRT_MAJOR_ENTERPRISE 10` line).
function(_trt_version_component out_var define_name version_file)
    file(STRINGS "${version_file}" _line REGEX "^#define ${define_name} .+$")
    string(REGEX REPLACE "^#define ${define_name} ([A-Za-z0-9_]+).*$" "\\1" _token "${_line}")
    if(_token MATCHES "^[0-9]+$")
        set(${out_var} "${_token}" PARENT_SCOPE)
    else()
        file(STRINGS "${version_file}" _indirect REGEX "^#define ${_token} [0-9]+.*$")
        string(REGEX REPLACE "^#define ${_token} ([0-9]+).*$" "\\1" _num "${_indirect}")
        set(${out_var} "${_num}" PARENT_SCOPE)
    endif()
endfunction()

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInfer.h")
    if(EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
        set(_VersionSearchFile "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    else()
        set(_VersionSearchFile "${TensorRT_INCLUDE_DIR}/NvInfer.h")
    endif()
    _trt_version_component(TensorRT_VERSION_MAJOR NV_TENSORRT_MAJOR "${_VersionSearchFile}")
    _trt_version_component(TensorRT_VERSION_MINOR NV_TENSORRT_MINOR "${_VersionSearchFile}")
    _trt_version_component(TensorRT_VERSION_PATCH NV_TENSORRT_PATCH "${_VersionSearchFile}")
    set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_LIBRARY TensorRT_INCLUDE_DIR
    VERSION_VAR TensorRT_VERSION_STRING
)

if(TensorRT_FOUND)
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})

    if(NOT TensorRT_LIBRARIES)
        set(TensorRT_LIBRARIES ${TensorRT_LIBRARY})
        if(TensorRT_nvinfer_plugin_LIBRARY)
            list(APPEND TensorRT_LIBRARIES ${TensorRT_nvinfer_plugin_LIBRARY})
        endif()
    endif()

    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT INTERFACE IMPORTED)
        add_library(TensorRT::TensorRT ALIAS TensorRT)
    endif()

    if(NOT TARGET TensorRT::nvinfer)
        add_library(TensorRT::nvinfer SHARED IMPORTED)
        set_target_properties(TensorRT::nvinfer PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${TensorRT_LIBRARY}"
        )
        target_link_libraries(TensorRT INTERFACE TensorRT::nvinfer)
    endif()

    if(NOT TARGET TensorRT::nvinfer_plugin AND TensorRT_nvinfer_plugin_LIBRARY)
        add_library(TensorRT::nvinfer_plugin SHARED IMPORTED)
        set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${TensorRT_nvinfer_plugin_LIBRARY}"
        )
        target_link_libraries(TensorRT INTERFACE TensorRT::nvinfer_plugin)
    endif()
endif()
