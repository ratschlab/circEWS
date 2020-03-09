import os
import logging


class Dataset(object):
    supported_versions = ['HiRID', 'MIMIC']

    def __init__(self, dataset_version='v5', overrides={}):
        if dataset_version not in self.supported_versions:
            raise ValueError('dataset_version of {} is invalid; must be one of {}'.format(dataset_version, ', '.join(self.supported_versions)))
        self.dataset_version = dataset_version

        if self.dataset_version == 'HiRID':
            selected_paths = self.HiRID_paths.copy()
        elif self.dataset_version == 'MIMIC':
            selected_paths = self.MIMIC_paths.copy()

        for key, value in overrides.items():
            if key in selected_paths.keys():
                logging.info('Overriding option {option_name} (default value: {option_default}) with {new_value}'.format(
                    option_name=key,
                    option_default=selected_paths[key],
                    new_value=value
                ))
                selected_paths[key] = value
            else:
                logging.warning('Dataset has no attribute {}, setting nevertheless...'.format(key))
                selected_paths[key] = value

        for key, value in selected_paths.items():
            setattr(self, key, value)

    HiRID_paths = {
        'merged_path': '../../../3_merged/',
        'endpointdir': '../../../3a_endpoints/',
        'imputed_variables_path': '../../../5_imputed/',
        'static_variables_path': '../../../5_imputed/static.h5',
        'labels_path': '../../../6_labels/',
        'split_path': '../../../misc_derived/temporal_split_180918.tsv',
        'X_y_path': '../../../7_ml_input/',
        'selected_variables': './misc/selected_variables.csv'
    }

    MIMIC_paths = {
        'merged_path': '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/merged/181023/reduced/',
        'endpointdir': '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/endpoints/181103/reduced/',
        'imputed_variables_path': '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/imputed/imputed_181023/reduced/held_out/',
        'static_variables_path': '',  # '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/5_imputed/imputed_180918/reduced/temporal_5/static.h5',
        'labels_path': '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/labels/targets_181023/',
        'split_path': '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/misc_derived/split_181015.pickle',
        'X_y_path': '/cluster/work/grlab/clinical/Inselspital/DataReleases/01-19-2017/InselSpital/external_validation/ml_input/181023/',
        'selected_variables': '/cluster/work/borgw/Bern_ICU_Sanctuary/v6b/selected_variables.csv'
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--get-setting', required=True, type=str)

    args = parser.parse_args()
    d = Dataset(args.dataset)
    print(getattr(d, args.get_setting))
