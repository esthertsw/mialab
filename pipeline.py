"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using a decision forest classifier.
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
from sklearn.model_selection import GridSearchCV
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

import matplotlib.pyplot as plt

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load

def save_slice(img, title, out_path, slice_idx=None):
    """Save a mid-axial slice of a SimpleITK 3D image."""
    #for debugging
    arr = sitk.GetArrayFromImage(img)
    if slice_idx is None:
        slice_idx = arr.shape[0] // 2  # axial middle slice
    plt.figure(figsize=(5, 5))
    plt.imshow(arr[slice_idx], cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

def count_voxels_per_class(image_list, label_key=structure.BrainImageTypes.GroundTruth):
    """Count the number of voxels per class over a list of images using a standard dict."""
    voxel_counts = {}
    for img in image_list:
        gt_img = img.images[label_key]
        gt_arr = sitk.GetArrayFromImage(gt_img)
        classes, counts = np.unique(gt_arr, return_counts=True)
        for c, count in zip(classes, counts):
            if c in voxel_counts:
                voxel_counts[c] += count
            else:
                voxel_counts[c] = count
    return voxel_counts

def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, random_forest_type: str):
    """Brain tissue segmentation using decision forests.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Feature extraction
        - Decision forest classifier model building
        - Segmentation using the decision forest classifier model on unseen images
        - Post-processing of the segmentation
        - Evaluation of the segmentation
    """
    # create a result directory with timestamp (moved to beginning to save debugging stuff in same dir)
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                        #   'brain_mask_morph': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True
                          }

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # =================================
    # Start of Pre-processing debugging
    # =================================
    # debug_dir = os.path.join(result_dir, "_debug_train")
    # os.makedirs(debug_dir, exist_ok=True)
    # example_img = images[0]
    # t1w_arr = sitk.GetArrayFromImage(example_img.images[structure.BrainImageTypes.T1w])
    # t2w_arr = sitk.GetArrayFromImage(example_img.images[structure.BrainImageTypes.T2w])
    # bm_arr = sitk.GetArrayFromImage(example_img.images[structure.BrainImageTypes.BrainMask])
    
    # print("example T1w shape", t1w_arr.shape)
    # print("example T2w shape",t2w_arr.shape)
    # print("example Brain mask shape", bm_arr.shape)
    
    # for img in images:
    #     save_slice(img.images[structure.BrainImageTypes.T1w],
    #             f"{img.id_} – T1 preproccessed",
    #             os.path.join(debug_dir, f"{img.id_}_T1pre.png"))

    #     save_slice(img.images[structure.BrainImageTypes.BrainMask],
    #             f"{img.id_} – Brain Mask",
    #             os.path.join(debug_dir, f"{img.id_}_mask.png"))

    #     save_slice(img.images[structure.BrainImageTypes.T2w],
    #             f"{img.id_} – T2 preproccessed",
    #             os.path.join(debug_dir, f"{img.id_}_T2pre.png"))
    # # =================================
    # End of Pre-processing debugging
    # =================================

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()

    # Couting voxel proportions
    train_voxel_counts = count_voxels_per_class(images)
    
    # =================================
    # Beginning Debugging features
    # =================================
    # example_img = images[0]
    # feat = example_img.feature_matrix[0]
    #
    # print("feature [0] shape", example_img.feature_matrix[0].shape)
    # print("feature [1] shape", example_img.feature_matrix[1].shape)
    # =================================
    # End Debugging features
    # =================================

    if random_forest_type=="GridSearch":
        print('-' * 5, 'Running GridSearch')
        params = {'n_estimators': [50, 75, 100, 125, 150], 'max_depth':[10, 20, 30, 40, 50]}
        model = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1])
        forest = GridSearchCV(model, params, cv=3)
    elif random_forest_type=="Balanced":
        #forest for class balanced version
        print('-' * 5, 'Using balanced random forest')
        forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1], 
                                                    n_estimators=100, 
                                                    max_depth=50, 
                                                    class_weight="balanced")
    elif random_forest_type=="Extra_imbalance":
        #forest for class balanced version
        print('-' * 5, 'Using exagerated imbalanced random forest')
        forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1], 
                                                    n_estimators=100, 
                                                    max_depth=50, 
                                                    class_weight={1:5, 2:5})
    
    elif random_forest_type=="Standard":
        forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1], 
                                                    n_estimators=100, 
                                                    max_depth=50)
    

    start_time = timeit.default_timer()
    forest.fit(data_train, labels_train)
    print(' Time elapsed:', timeit.default_timer() - start_time, 's')
    #print(' GridSearch best parameters: ', forest.best_params_)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    #Counting voxels for metric eval
    test_voxel_counts = count_voxels_per_class(images_test)
    
    #saving voxel counts in csv
    all_classes = set(train_voxel_counts.keys()).union(test_voxel_counts.keys())
    total_train = sum(train_voxel_counts.values())
    total_test = sum(test_voxel_counts.values())
    total_all = total_train + total_test

    csv_rows = []
    for cls in sorted(all_classes):
        train_count = train_voxel_counts.get(cls, 0)
        test_count = test_voxel_counts.get(cls, 0)
        total_count = train_count + test_count
        csv_rows.append({
            'Class': cls,
            'Training_Voxels': train_count,
            'Testing_Voxels': test_count,
            'Total_Voxels': total_count,
            'Training_%': train_count / total_train * 100 if total_train > 0 else 0,
            'Testing_%': test_count / total_test * 100 if total_test > 0 else 0,
            'Total_%': total_count / total_all * 100 if total_all > 0 else 0
        })

    header = ['Class', 'Training_Voxels', 'Testing_Voxels', 'Total_Voxels',
          'Training_%', 'Testing_%', 'Total_%']

    csv_file = os.path.join(result_dir, 'voxel_counts_summary.csv')

    with open(csv_file, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in csv_rows:
            line = [str(row[h]) for h in header]
            f.write(','.join(line) + '\n')
    # end of voxel count saving for evaluation    

    images_prediction = []
    images_probabilities = []

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)
 
        start_time = timeit.default_timer()
        predictions = forest.predict(img.feature_matrix[0])
        probabilities = forest.predict_proba(img.feature_matrix[0])
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)

    # post-process segmentation and evaluate with post-processing
    post_process_params = {'simple_post': True, 'morph_radius': 0, 'min_size': 50}  # morph_radius of 0 makes post-processing diff much smaller (1M to 10k)
    # post_process_params = {'crf_post': True}
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities,
                                                     post_process_params, multi_process=False)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],
                           img.id_ + '-PP')

        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)
        sitk.WriteImage(img.images[structure.BrainImageTypes.GroundTruth], os.path.join(result_dir, images_test[i].id_ + '_GT_reg.mha'), True)
        sitk.WriteImage(img.images[structure.BrainImageTypes.T1w], os.path.join(result_dir, images_test[i].id_ + '_T1w_reg.mha'), True)
        sitk.WriteImage(img.images[structure.BrainImageTypes.T2w], os.path.join(result_dir, images_test[i].id_ + '_T2w_reg.mha'), True)        

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, 'mialab/data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, 'mialab/data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, 'mialab/data/test/')),
        help='Directory with testing data.'
    )

    parser.add_argument(
        '--RF',
        type=str,
        default="Standard",
        help='Specify whether to run random forest with GridSearch or Balanced classes'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, args.RF)
