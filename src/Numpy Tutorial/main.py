from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator
def main():
    # Use Checker
    checker = Checker(resolution=100, size=10)
    checker.draw()
    checker.show()

    # Use Circle
    circle = Circle(resolution=100, radius=30, position=(50, 50))
    circle.draw()
    circle.show()

    # Use Spectrum
    spectrum = Spectrum(resolution=100)
    spectrum.draw()
    spectrum.show()

    # Use ImageGenerator
    file_path = "/Users/prashanthgadwala/Documents/Study material/Semester2/Deep learning/Exercise/exercise0_material/src_to_implement/exercise_data/"
    label_path = "/Users/prashanthgadwala/Documents/Study material/Semester2/Deep learning/Exercise/exercise0_material/src_to_implement/Labels.json"
    batch_size = 6
    image_size = (36, 36)
    generator = ImageGenerator(file_path, label_path, batch_size, image_size, rotation=True, mirroring=True)
    generator.show()

if __name__ == "__main__":
    main()