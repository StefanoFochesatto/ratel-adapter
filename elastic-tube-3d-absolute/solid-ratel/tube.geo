// Gmsh script for 3D Elastic Tube
// Style adapted from Ratel's hollow-cylinder.geo

DefineConstant[
  r_i = {0.005, Name "Parameters/r_i"}
  r_o = {0.006, Name "Parameters/r_o"}
  extrude_length = {0.05, Name "Parameters/extrusion length"}
  extrude_layers = {80, Name "Parameters/extrusion layers"}
  radial_elements = {2, Name "Parameters/radial elements"}
  quadrant_elements = {6, Name "Parameters/quadrant elements"}
];

Point(1) = {0, 0, 0};

Point(2) = {r_i, 0, 0};
Point(3) = {0, r_i, 0};
Point(4) = {-r_i, 0, 0};
Point(5) = {0, -r_i, 0};

Point(6) = {r_o, 0, 0};
Point(7) = {0, r_o, 0};
Point(8) = {-r_o, 0, 0};
Point(9) = {0, -r_o, 0};

Circle(12) = {2, 1, 3};
Circle(13) = {3, 1, 4};
Circle(14) = {4, 1, 5};
Circle(15) = {5, 1, 2};
Circle(16) = {6, 1, 7};
Circle(17) = {7, 1, 8};
Circle(18) = {8, 1, 9};
Circle(19) = {9, 1, 6};

Line(21) = {2, 6};
Line(22) = {3, 7};
Line(23) = {4, 8};
Line(24) = {5, 9};

Curve Loop(1) = {21, 16, -22, -12};
Curve Loop(2) = {22, 17, -23, -13};
Curve Loop(3) = {23, 18, -24, -14};
Curve Loop(4) = {24, 19, -21, -15};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};

Recombine Surface {1:4};
Transfinite Curve {21:24} = radial_elements + 1;
Transfinite Curve {12:19} = quadrant_elements + 1;
Transfinite Surface {1:4};

// Extrude into 3D volume
vol[] = Extrude {0, 0, extrude_length} { Surface{1:4}; Layers{extrude_layers}; Recombine; };

// Identify surfaces for Physical Groups
// start faces are 1, 2, 3, 4
// end faces are vol[0], vol[6], vol[12], vol[18]
// inner surfaces correspond to the 4th line in each loop (-12, -13, -14, -15)
// vol[] indices: [top, volume, side1, side2, side3, side4, ...]
Physical Surface("Clamped", 1) = {1, 2, 3, 4, vol[0], vol[6], vol[12], vol[18]};
Physical Surface("Coupling", 2) = {vol[5], vol[11], vol[17], vol[23]};
Physical Volume("TubeVolume", 3) = {vol[1], vol[7], vol[13], vol[19]};

Mesh.ElementOrder = 1;
Mesh.MshFileVersion = 2.2;
