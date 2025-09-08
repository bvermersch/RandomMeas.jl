# Directory helper functions
src_dir() = dirname(pathof(@__MODULE__))
pkg_dir() = joinpath(dirname(pathof(@__MODULE__)), "..")

# Determine version and uuid of the package
function _parse_project_toml(field::String)
  return Pkg.TOML.parsefile(joinpath(pkg_dir(), "Project.toml"))[field]
end
version() = VersionNumber(_parse_project_toml("version"))
uuid() = Base.UUID(_parse_project_toml("uuid"))
