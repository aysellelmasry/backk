{ pkgs }:
{
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.cmake
    pkgs.dlib
    pkgs.libGL
    pkgs.libGLU
    pkgs.glib
  ];
}
