# Lab task: the focal length of your own camera

Session of 13 November 2026 · group task · about 45 minutes

## Goal

Every team finds the focal length **f** of a laptop webcam, in pixels, and
writes it on the board. Then we compare.

## Team checkerboard

For teams with a second screen (laptop or tablet).

1. Show `data/checkerboard_9x6.png` full-screen on the second screen.
   Brightness up, no reflections.
2. Measure the full width of the board (10 squares) in millimetres and
   divide by 10. That is your square size. Do not zoom the image afterwards.
3. From the `src/` folder, capture 10 to 15 views. Tilt and move the screen
   or the camera between shots. Space keeps a view, q ends.

   ```
   python camera_calibration.py capture --out myboard
   ```

4. Solve, with your own square size in mm:

   ```
   python camera_calibration.py solve --images myboard --square 20
   ```

5. Note fx, fy and the reprojection error. Below 1 px is good.

## Team tissue pack

For teams without a second screen. No script needed.

1. Measure the long side of a tissue pack with a ruler: **Y**, in mm.
2. Stand it up at a measured distance from the webcam, for example 500 mm:
   **Z**.
3. Take a photo and read off the height of the pack in pixels: **y**.
4. Compute the focal length:

   ```
   f = y × Z ÷ Y
   ```

5. Test it. Move the pack to a distance you have not measured, take a new
   photo, and predict the distance:

   ```
   Z = f × Y ÷ y
   ```

   Then check with the ruler. How far off are you?

## To discuss afterwards

- Do the two methods agree on f for the same laptop model?
- What does the checkerboard give you that the tissue pack cannot?
- What happens to f if the webcam records at a different resolution?
