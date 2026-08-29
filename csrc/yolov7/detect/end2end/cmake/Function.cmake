
function(print_var var)
    set(value "${${var}}")
    string(LENGTH "${value}" value_length)
    if(value_length GREATER 0)
        math(EXPR last_index "${value_length} - 1")
        string(SUBSTRING "${value}" ${last_index} ${last_index} last_char)
    endif()

    if(NOT "${last_char}" STREQUAL "\n")
        set(value "${value}\n")
    endif()
    message(STATUS "${var}:\n   ${value}")
endfunction()
