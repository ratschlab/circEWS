import pandas as pd
import unittest


class TestDynamic(unittest.TestCase):
    def test_control_patient_processing(self):
        from utils.mock_data import mock_no_endpoint_switch
        from shapelets.extract_endpoint_times_dynamic import process_patient

        patient_id = 42
        duration = pd.DateOffset(hours=1)
        dt = pd.DateOffset(days=0.5)  # This value should not matter
        # Due to randomness, lets do some repititions so we can be pretty certain everything is ok
        for i in range(100):
            results = process_patient(patient_id, mock_no_endpoint_switch, duration, dt, None)
            if not isinstance(results, dict):
                print(results)
            self.assertIsInstance(results, dict)
            extracted = results[str(patient_id)]
            self.assertLessEqual(extracted['Start'][0], pd.Timestamp('12/1/2018 12:55:00'))
            self.assertGreaterEqual(extracted['End'][0], pd.Timestamp('12/1/2018 12:55:00'))
            self.assertEqual(extracted['s3m_endpoint'][0], 0)

    def test_case_patient_processing(self):
        from utils.mock_data import mock_endpoint_switch
        from shapelets.extract_endpoint_times_dynamic import process_patient

        patient_id = 42
        duration = pd.DateOffset(hours=1)
        dt = pd.DateOffset(hours=0.5)
        results = process_patient(patient_id, mock_endpoint_switch, duration, dt, None)
        if not isinstance(results, dict):
            print(results)

        self.assertIsInstance(results, dict)
        extracted = results[str(patient_id)]
        self.assertEqual(extracted['Start'][0], pd.Timestamp('12/1/2018 12:00:00'))
        self.assertEqual(extracted['End'][0], pd.Timestamp('12/1/2018 13:00:00'))
        self.assertEqual(extracted['s3m_endpoint'][0], 1)

    def test_insufficient_data_processing(self):
        from utils.mock_data import mock_insufficient_data
        from shapelets.extract_endpoint_times_dynamic import process_patient
        from utils.mp import ProcessingError

        patient_id = 42
        duration = pd.DateOffset(hours=1)
        dt = pd.DateOffset(hours=0.5)
        results = process_patient(patient_id, mock_insufficient_data, duration, dt, None)

        self.assertIsInstance(results, ProcessingError)
        self.assertEqual(results.errortype, ProcessingError.Type.InsufficientData)

    def test_no_valid_endpoints_processing(self):
        from utils.mock_data import mock_no_valid_endpoint
        from shapelets.extract_endpoint_times_dynamic import process_patient
        from utils.mp import ProcessingError

        patient_id = 42
        duration = pd.DateOffset(hours=1)
        dt = pd.DateOffset(hours=0.5)
        results = process_patient(patient_id, mock_no_valid_endpoint, duration, dt, None)

        self.assertIsInstance(results, ProcessingError)
        self.assertEqual(results.errortype, ProcessingError.Type.NoValidEndpoint)

    def test_NaN_handling_control(self):
        from utils.mock_data import mock_control_contains_NaNs
        from shapelets.extract_endpoint_times_dynamic import process_patient

        patient_id = 42
        duration = pd.DateOffset(hours=0.25)
        dt = pd.DateOffset(hours=0.25)
        results = process_patient(patient_id, mock_control_contains_NaNs, duration, dt, None)

        self.assertEqual(len(results), 1)

        extracted = results[str(patient_id)]

        self.assertGreaterEqual(extracted['Start'][0], pd.Timestamp('12/1/2018 13:00:00'))
        self.assertLessEqual(extracted['End'][0], pd.Timestamp('12/1/2018 13:55:00'))
        self.assertEqual(extracted['s3m_endpoint'][0], 0)

    def test_NaN_handling_case(self):
        from utils.mock_data import mock_case_contains_NaNs
        from shapelets.extract_endpoint_times_dynamic import process_patient

        patient_id = 42
        duration = pd.DateOffset(hours=0.25)
        dt = pd.DateOffset(hours=0.25)
        results = process_patient(patient_id, mock_case_contains_NaNs, duration, dt, None)

        self.assertEqual(len(results), 1)

        extracted = results[str(patient_id)]

        self.assertEqual(extracted['Start'][0], pd.Timestamp('12/1/2018 12:35:00'))
        self.assertEqual(extracted['End'][0], pd.Timestamp('12/1/2018 12:50:00'))
        self.assertEqual(extracted['s3m_endpoint'][0], 1)

class TestStatic(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
