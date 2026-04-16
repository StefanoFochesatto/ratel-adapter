// Gmsh script for perpendicular flap
W = 0.1;
H = 1.0;

Point(1) = {-W/2, 0, 0, 0.025};
Point(2) = { W/2, 0, 0, 0.025};
Point(3) = { W/2, H, 0, 0.025};
Point(4) = {-W/2, H, 0, 0.025};

Line(1) = {1, 2}; // Bottom (Clamped)
Line(2) = {2, 3}; // Right (Coupling)
Line(3) = {3, 4}; // Top (Coupling)
Line(4) = {4, 1}; // Left (Coupling)

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Labels for Ratel/PETSc
Physical Curve("Clamped") = {1};
Physical Curve("Coupling") = {2, 3, 4};
Physical Surface("Flap") = {1};

// Optional: Transfinite mesh for structured elements
Transfinite Line {1, 3} = 5; // 4 elements across width
Transfinite Line {2, 4} = 41; // 40 elements along height
Transfinite Surface {1};
Recombine Surface {1}; // Use quads
