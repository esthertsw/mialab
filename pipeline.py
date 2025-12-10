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
import pickle

import matplotlib.pyplot as plt

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
    import mialab.utilities.metric_tricks as mutil
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

def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str, random_forest_type: str, run_metric_tricks: bool, load_model:bool, save_model_weights:bool):
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
    
    if not load_model or not f'{random_forest_type.lower()}_model.pkl' in os.listdir('weights'): # store weights if not available
        print("Training model")
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
        elif random_forest_type=="Weighted_large":
            #forest for class balanced version
            print('-' * 5, 'Using random forest weighted to large classes')
            forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1], 
                                                        n_estimators=100, 
                                                        max_depth=50, 
                                                        class_weight={1:10, 2:10})
        elif random_forest_type=="Weighted_small":
            #forest for class balanced version
            print('-' * 5, 'Using random forest weighted to small classes')
            forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1], 
                                                        n_estimators=100, 
                                                        max_depth=50, 
                                                        class_weight={3:5, 4:5, 5:5})
        
        elif random_forest_type=="Standard":
            forest = sk_ensemble.RandomForestClassifier(max_features=images[0].feature_matrix[0].shape[1], 
                                                        n_estimators=150, 
                                                        max_depth=40)
        

        start_time = timeit.default_timer()
        forest.fit(data_train, labels_train)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')
        #print(' GridSearch best parameters: ', forest.best_params_)

        if save_model_weights:
            with open(f'weights/{random_forest_type.lower()}_model.pkl','wb') as f:
                pickle.dump(forest,f)
            print(random_forest_type, ' Model saved.')
    elif load_model: 
        with open(f'weights/{random_forest_type.lower()}_model.pkl','rb') as f:
            forest = pickle.load(f)
        print(random_forest_type, ' Model loaded.')

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
        
        # Run metric-trick experiments (OPTIONAL)
        
        if run_metric_tricks:


            tricks_out = os.path.join(result_dir, "metric_tricks")
            os.makedirs(tricks_out, exist_ok=True)

            print(f"Running metric-trick experiments for {img.id_} ...")

            seg_pp = images_post_processed[i]
            gt_reg = img.images[structure.BrainImageTypes.GroundTruth]

            # Trick 1: Largest CC only
            manipulated_lcc = mutil.run_metric_trick_experiment(
                seg_pp, gt_reg, tricks_out,
                f"{img.id_}_largestCC",
                mutil.keep_largest_cc
            )
            evaluator.evaluate(
                manipulated_lcc, gt_reg,
                img.id_ + "-TRICK-largestCC"
            )

            # Trick 2: Shrink boundary
            manipulated_shrink = mutil.run_metric_trick_experiment(
                seg_pp, gt_reg, tricks_out,
                f"{img.id_}_shrink",
                lambda x: mutil.shrink_boundary(x, radius=0.5)
            )
            evaluator.evaluate(
                manipulated_shrink, gt_reg,
                img.id_ + "-TRICK-shrink"
            )

            # Trick 3: Remove far voxels
            manipulated_remove = mutil.run_metric_trick_experiment(
                seg_pp, gt_reg, tricks_out,
                f"{img.id_}_removeDist",
                lambda x: mutil.remove_far_voxels( x,
                            frac_per_label={  # tune these if you want!
                                1: 0.95,  # WM  keep 95% of radial extent
                                2: 0.95,  # GM
                                3: 0.70,  # Hippocampus
                                4: 0.70,  # Amygdala
                                5: 0.80,  # Thalamus
                            },
                    default_frac=0.8
                )
            )   
            evaluator.evaluate(
                manipulated_remove, gt_reg,
                img.id_ + "-TRICK-removeDist"
            )

            # Trick 4: Morphological closing on labeled voxels
            manipulated_closed = mutil.run_metric_trick_experiment(
                seg_pp, gt_reg, tricks_out,
                f"{img.id_}_morphClose",
                lambda x: mutil.mask_dilation_and_erosion(x, result_dir=None, img_id=None) # NOTE result_dir and img.id_ required only if you want to save the img after labels are changed
            ) 
            evaluator.evaluate(
                manipulated_closed, gt_reg,
                img.id_ + "-TRICK-morphClose"
            )

        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)
        sitk.WriteImage(img.images[structure.BrainImageTypes.GroundTruth], os.path.join(result_dir, images_test[i].id_ + '_GT_reg.mha'), True)
        sitk.WriteImage(img.images[structure.BrainImageTypes.T1w], os.path.join(result_dir, images_test[i].id_ + '_T1w_reg.mha'), True)
        sitk.WriteImage(img.images[structure.BrainImageTypes.T2w], os.path.join(result_dir, images_test[i].id_ + '_T2w_reg.mha'), True)        

    # use two writers to report the results
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

    # Generate global trick summary plots
    if run_metric_tricks:
        tricks_dir = os.path.join(result_dir, "metric_tricks")

        print("\nGenerating trick-summary boxplots...")

        # Load all trick CSV results into a dataframe
        df = mutil.load_trick_results(tricks_dir)

        # Create summary plots (one figure per trick)
        mutil.plot_trick_summary_boxplot(df, tricks_dir)

        print("Trick summary plots saved in:", tricks_dir)
        
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

    parser.add_argument(
        '--load_model_weights',
        action='store_true',
        help='Include if stored model should be used (if available) instead of training'
    )

    parser.add_argument(
        '--save_model_weights',
        action='store_true',
        help='Save model weights after training'
    )
    
    parser.add_argument(
        '--run_metric_tricks',
        action='store_true',
        help='Run metric manipulation experiments (largest CC, shrink, distance trimming)'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir, args.RF, args.run_metric_tricks, args.load_model_weights, args.save_model_weights)
