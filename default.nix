let
     pkgs = import <nixpkgs> {};
     stdenv = pkgs.stdenv;
in rec {
          UCLATCHEnv = stdenv.mkDerivation rec {
          name = "UCLATCH-env";
          version = "1.1.1.1";
          src = ./.;
          buildInputs = [ pkgs.opencv3 pkgs.cudatoolkit pkgs.gnumake pkgs.pkgconfig pkgs.stdenv pkgs.gcc49 pkgs.cmake ];
     };
}
