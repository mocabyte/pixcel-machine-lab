# Lab task: the focal length of your phone camera

Session of 13 November 2026 · individual task · about 45 minutes

## Goal

You find the focal length **f** of your own phone camera, in pixels, and
write it on the board next to your phone model. Then we compare.

## Three rules, or it will not work

1. **Landscape only, 1x lens, no zoom.** Every photo the same way. Mixed
   orientations or lenses break the result.
2. **JPG, not HEIC.** OpenCV cannot read iPhone HEIC files. Export them as
   JPG first.
3. **The whole board in every photo.** If a corner is cut off, the photo is
   useless.

## What you need

- Your laptop, showing the checkerboard
- Your phone, taking the photos
- A ruler, or the Measure app

## Steps

1. **Show the board.** Open `data/checkerboard_9x6.png` full-screen on your
   laptop. Brightness up, no reflections on the screen.

2. **Measure a square.** Measure the full width of the board (10 squares) in
   millimetres and divide by 10. Do not zoom the image after measuring.

3. **Take 10 to 15 photos with your phone.**
   - Always landscape, always the normal 1x lens, no zoom.
   - The whole board must be visible in every photo.
   - Change the view each time: from the left, from the right, from above,
     closer, further away, board in a corner of the picture.
   - Fifteen different views beat fifty identical ones.

4. **Move the photos to your laptop** into a new folder `src/myboard/`
   (AirDrop, cable or cloud). They must be JPG. On an iPhone, AirDrop
   usually converts them; if you end up with `.heic` files, export them as
   JPG first.

5. **Solve.** From the `src/` folder, with your own square size in mm:

   ```
   python camera_calibration.py solve --images myboard --square 20
   ```

6. **Read the result.**
   - `fx`, `fy`: the focal length in pixels. They should be nearly equal.
   - `cx`, `cy`: the image centre. Close to half the width and height?
   - `reprojection error`: below 1 px is good.
   - If the error is high, remove the worst view the script names and solve
     again.

## Check it against the world

Take one more photo of an object whose height you know, for example a
tissue pack, from a distance you measured. Read off its height in pixels
and predict the distance:

```
Z = f × Y ÷ y
```

Y is the real height in mm, y the height in the photo in pixels. How far
off are you?

## To discuss afterwards

- Why is your f so much larger than the 800 px from the lecture?
- Do two phones of the same model get the same f?
- What happens to f if you switch to the wide-angle or the tele lens?
