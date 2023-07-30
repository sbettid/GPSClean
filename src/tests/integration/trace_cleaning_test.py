from gpsclean.main import main
import os

class TestTraceCleaning:
    # test input traces
    test_input_trace_outliers_basepath = "./src/tests/integration/inputs/test_input_trace_outliers"
    test_input_trace_pauses_basepath = "./src/tests/integration/inputs/test_input_trace_outliers"
    
    test_input_trace_outliers = f"{test_input_trace_outliers_basepath}.gpx"
    test_input_trace_pauses = f"{test_input_trace_pauses_basepath}.gpx"

    # expected outputs pauses
    expected_outputs_pauses = [
        f"{test_input_trace_pauses_basepath}_cleaned.gpx",
        f"{test_input_trace_pauses_basepath}_predicted.geojson",
        f"{test_input_trace_pauses_basepath}_predictedColors.geojson"
    ]

    # expected outputs outliers
    expected_outputs_outliers = [
        f"{test_input_trace_outliers_basepath}_cleaned.gpx",
        f"{test_input_trace_outliers_basepath}_predicted.geojson",
        f"{test_input_trace_outliers_basepath}_predictedColors.geojson"
    ]

    @classmethod
    def remove_file_if_exists(self, filepath):
        if os.path.exists(filepath):
            os.remove(filepath)

    @classmethod
    def trace_cleaning(self, trace, expected_outputs):
        # try to clean the test trace
        main([trace, "--outputPredictions", "--meanPredictionColored"])
        
        # check the cleaned trace, predictions files, and mean colored predictions file were created
        for expected_file in expected_outputs:
            assert os.path.exists(expected_file) == True
            self.remove_file_if_exists(expected_file)

    def test_cleaning_pauses_full_output(self):
        self.trace_cleaning(self.test_input_trace_pauses, self.expected_outputs_pauses)

    def test_cleaning_outliers_full_output(self):
        self.trace_cleaning(self.test_input_trace_outliers, self.expected_outputs_outliers)
