// beam_tetra.geo - 3D Tetrahedral Beam
// Dimensions
L = 10.0; W = 2.0; D = 2.0;
lc = 1.0; // Characteristic length (set mesh density)

// Bottom Points (Z=0)
Point(1) = {0, 0, 0, lc};
Point(2) = {W, 0, 0, lc};
Point(3) = {W, D, 0, lc};
Point(4) = {0, D, 0, lc};

// Top Points (Z=L)
Point(5) = {0, 0, L, lc};
Point(6) = {W, 0, L, lc};
Point(7) = {W, D, L, lc};
Point(8) = {0, D, L, lc};

// Bottom Edges
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
// Top Edges
Line(5) = {5, 6}; Line(6) = {6, 7}; Line(7) = {7, 8}; Line(8) = {8, 5};
// Vertical Edges
Line(9) = {1, 5}; Line(10) = {2, 6}; Line(11) = {3, 7}; Line(12) = {4, 8};

// 6 Surfaces (Faces)
Curve Loop(1) = {1, 2, 3, 4};     Plane Surface(1) = {1}; // Bottom (Z=0)
Curve Loop(2) = {5, 6, 7, 8};     Plane Surface(2) = {2}; // Top (Z=L)
Curve Loop(3) = {1, 10, -5, -9};  Plane Surface(3) = {3}; // Front (Y=0)
Curve Loop(4) = {2, 11, -6, -10}; Plane Surface(4) = {4}; // Right (X=W)
Curve Loop(5) = {3, 12, -7, -11}; Plane Surface(5) = {5}; // Back (Y=D)
Curve Loop(6) = {4, 9, -8, -12};  Plane Surface(6) = {6}; // Left (X=0)

// 1 Solid Volume
// Surface Loop defines the shell, Volume defines the interior
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

// Physical Groups for PETSc
// These define the Labels and IDs
Physical Surface("FixedBottom", 1) = {1};
Physical Surface("TractionSurface", 2) = {2, 3, 4, 5, 6};
Physical Volume("BeamVolume", 3) = {1};
