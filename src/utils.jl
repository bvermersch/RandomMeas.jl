# Copyright (c) 2025 Benoît Vermersch and Andreas Elben
# SPDX-License-Identifier: Apache-2.0
# http://www.apache.org/licenses/LICENSE-2.0


"""
RandomMeas.jl - Utility Functions

This module provides utility functions for package management and directory operations.
"""

"""
    src_dir()

Get the source directory path of the current module.

# Returns
A string containing the path to the source directory.
"""
src_dir() = dirname(pathof(@__MODULE__))

"""
    pkg_dir()

Get the package root directory path.

# Returns
A string containing the path to the package root directory.
"""
pkg_dir() = joinpath(dirname(pathof(@__MODULE__)), "..")

"""
    _parse_project_toml(field::String)

Parse a specific field from the Project.toml file.

# Arguments
- `field::String`: The field name to extract from Project.toml.

# Returns
The value of the specified field from Project.toml.
"""
function _parse_project_toml(field::String)
  return Pkg.TOML.parsefile(joinpath(pkg_dir(), "Project.toml"))[field]
end

"""
    version()

Get the version number of the package.

# Returns
A `VersionNumber` representing the package version.
"""
version() = VersionNumber(_parse_project_toml("version"))

"""
    uuid()

Get the UUID of the package.

# Returns
A `Base.UUID` representing the package UUID.
"""
uuid() = Base.UUID(_parse_project_toml("uuid"))
