from gpsclean.gpsclean import main
import os

class TestTraceCleaning:
    # test input trace
    test_input_trace_basepath = "./src/tests/integration/inputs/test_input_trace"
    test_input_trace = f"{test_input_trace_basepath}.gpx"

    # expected outputs
    expected_cleaned_trace_path = f"{test_input_trace_basepath}_cleaned.gpx"
    expected_predictions_path = f"{test_input_trace_basepath}_predicted.geojson"
    expected_colored_predictions_path = f"{test_input_trace_basepath}_predictedColors.geojson"

    @classmethod
    def remove_file_if_exists(self, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)

    @classmethod
    def teardown_class(self):
        # teardown: remove output files, if existing
        self.remove_file_if_exists(self.expected_cleaned_trace_path)
        self.remove_file_if_exists(self.expected_predictions_path)
        self.remove_file_if_exists(self.expected_colored_predictions_path)

    def test_cleaning_full_output(self):

        # try to clean the test trace
        main([self.test_input_trace, "--outputPredictions", "--meanPredictionColored"])
        
        # check the cleaned trace, predictions files, and mean colored predictions file were created
        assert os.path.exists(self.expected_cleaned_trace_path) == True
        assert os.path.exists(self.expected_predictions_path) == True
        assert os.path.exists(self.expected_colored_predictions_path) == True
