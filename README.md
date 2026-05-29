🕸️ Bio-Inspired Spider Web Topology Optimization
Welcome! This code is a Topology Optimization (TopOpt) algorithm. Its goal is to design a structure that acts like a spider web. You give it a limited budget of "material," tell it where the loads (impacts) will hit, and the algorithm mathematically "grows" the most efficient, energy-absorbing net possible to catch those loads.

This README is written in plain English. No advanced physics degree is required to use or tweak this code.

🚀 Quick Start: How to Run It
1. Prerequisites
Before running the code, you need a few standard Python libraries installed. Open your terminal or command prompt and run:
pip install numpy scipy matplotlib numba

You also must ensure you have these two files in the exact same folder as the main script:

mma.py: The mathematical solver (Method of Moving Asymptotes).

polar_kinematics.py: The physics definitions.

2. Run the Script
Simply run the script in your terminal or IDE:
python NonLinearPlate_TopOpt_Polar.py

A live window will pop up showing the web "growing" in black and white. Once it finishes (around 150 iterations), it will generate a final 3D visualization of how the web sags under load.

🎛️ The Configuration Knobs (How to Tweak It)
At the top of the script, there is a configuration block. These are the levers you can pull to change the outcome.

NELR, NELT_SECTOR, N_SLICES: The grid resolution. Higher numbers mean a prettier, more detailed web, but it will take much longer to calculate.

VOLFRAC (Volume Fraction): Your material budget. 0.35 means the final web can only use 35% of the total available canvas space. The rest must be empty air.

PENAL (Penalty): Forces the algorithm to choose between pure black (solid) or pure white (empty space). If your web looks like a blurry gray cloud, increase this to 4.0 or 5.0 to force crisp lines.

R_MIN (Filter Radius): The thickness of the web strands. Lowering this makes the spider silk thinner. Warning: If you drop it below 0.3, the math might glitch out.

EPS_FAIL (Strain Limit): How far the web is allowed to stretch before it "snaps." Increasing this (e.g., 0.50) makes the web incredibly deep and stretchy.

🧠 How the Algorithm Actually Works (The Big Picture)
Imagine you have a piece of graphing paper, and you are playing a game with a robot.

The Setup: You tell the robot, "I am going to drop weights in these specific areas. You have 35% of the paper's area to draw lines to support them."

The Physics (Forward Solve): The robot guesses a design, applies the weights, and calculates exactly how much the paper stretches and deforms.

The Sensitivities (Adjoint Solve): The robot looks at the stretched paper and asks, "If I add a little more ink here, or remove some there, how much stiffer does the web get?"

The Update (MMA): The robot takes that information and draws a slightly better design.

Repeat: It loops this process until the web stops improving.

🔍 Function-by-Function Breakdown
Here is exactly what each block of code is doing, translated from math into English.

precompute_cyclic_filter
What it does: Prevents the "Checkerboard Bug".
Plain English: Algorithms are notorious for cheating. If left alone, this algorithm will try to build a structure out of alternating black and white pixels (a checkerboard) because the finite-element math sometimes miscalculates that pattern as infinitely stiff. This function creates a "blur" filter that forces neighboring pixels to blend smoothly, making checkerboarding impossible. The "cyclic" part just means it wraps perfectly in a 360-degree circle.

precompute_isoparametric_data & compute_G_and_weight
What it does: Sets up the geometry math for the physics engine.
Plain English: The web is made up of hundreds of tiny 4-sided shapes (elements). To figure out how these shapes stretch and bend, the computer needs to calculate transformation matrices (converting squares into trapezoids). This function pre-calculates all that annoying geometry math so the physics engine doesn't have to do it from scratch every single loop.

assemble_system_numba
What it does: The Core Physics Engine.
Plain English: This is the heavy lifter. The @njit tag at the top uses a library called Numba to turn Python into blazing-fast C-code. This function looks at the current design (where the material is) and figures out two things:

Internal Forces: How hard the material is fighting back against the weights.

Stiffness Matrix: How rigid the entire structure is.
It uses non-linear kinematics, which is a fancy way of saying it understands the "Rubber Sheet Effect." It knows that as a flat net deflects downward, it stretches like a drumhead, getting tighter and stiffer.

main
What it does: The Director that runs the show.
Plain English: This is where the magic happens. It follows these exact steps:

Draws the Canvas: Creates the circular grid (r_vec, t_vec, node_x).

Sets the Rules: Glues the outer edge of the web to the wall (fixed_dofs), sets up the Catchment Rings where the weights will hit (target_ring_radii), and locks those ring locations as solid material (passive_solid).

Starts the Loop: Enters the while loop (up to MAX_ITER).

Simulates: Pushes down on the web in 4 gradual increments (num_load_steps) to see how it flexes.

Evaluates: Calculates the "Minimum Compliance" (a mathematical way to measure how efficiently the web transfers the falling weight to the walls without wasting energy).

Updates the Design: Passes the data to Svanberg's MMA, which mathematically nudges the black and white pixels around to make the web slightly better.

Draws the Picture: Updates the live preview window.

🛠️ Troubleshooting & FAQ
Why does the web look like a blurry gray cloud?
The algorithm is realizing that a 40%-dense gray area acts like a highly flexible rubber sheet, which absorbs energy really well. If you want crisp, real-world threads, increase the PENAL value from 3.0 to 4.0.

Why did the program crash with a linear algebra error?
If VOLFRAC (your material budget) is too low, the algorithm physically cannot afford to connect the heavy rings to the wall. The physics engine breaks because the loads are just floating in mid-air. Increase VOLFRAC.

Why are there straight lines going outward (spokes)?
Because we told it to! Look for passive_solid[er * NELT_SECTOR + 0] = True in the main function. This hard-codes one solid radial line per slice to give the web a foundational skeleton.