add_rules("mode.debug", "mode.release")

set_optimize("fastest")
set_toolchains("mingw")
set_languages("c++latest")
add_rules("plugin.compile_commands.autoupdate")

target("GA")
    set_kind("binary")
    add_files("src/*.cpp")