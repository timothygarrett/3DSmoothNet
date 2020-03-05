/**
3DSmoothNet
main.cpp

Purpose: executes the computation of the SDV voxel grid for the selected interes points

@Author : Zan Gojcic, Caifa Zhou
@Version : 1.0
*/

// STL
#include <chrono>
#include <string>
#include <vector>

// PCL
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>

// Local
#include "core/core.h"
#include <sstream>

int main(int argc, char *argv[])
{
    // Turn off the warnings of pcl
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    // Initialize the variables
    std::string data_file;
    float radius;
    int num_voxels;
    float smoothing_kernel_width;
    std::string interest_points_file;
    std::string output_folder;
    std::string output_file;
    int chunk_size;

    // Get command line arguments
    bool result = processCommandLine(argc, argv, data_file, radius, num_voxels, smoothing_kernel_width, interest_points_file, 
                                     output_folder, output_file, chunk_size);
    if (!result)
        return 1;

    // Check if the output folder exists, otherwise create it
    boost::filesystem::path dir(output_folder);
    if (boost::filesystem::create_directory(dir))
    {
        std::cerr << "Directory Created: " << output_folder << std::endl;
    }

    // Read in the point cloud using the ply reader
    std::cout << "Config parameters successfully read in!! \n" << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::string ext = data_file.substr(data_file.length() - 4, 4);
    if (!fileExist(data_file)) {
        std::cerr << "Point cloud file does not exist or cannot be opened!!" << std::endl;
        return -1;
    }
    else if (ext.compare(".pcd") == 0 || ext.compare(".PCD") == 0)
        pcl::io::loadPCDFile(data_file, *cloud);
    else if (ext.compare(".ply") == 0 || ext.compare(".PLY") == 0)
        pcl::io::loadPLYFile(data_file, *cloud);
    else {
        std::cerr << "Invalid point cloud file type" << std::endl;
        return -1;
    }

    std::cout << "File: " << data_file << std::endl;
    std::cout << "Number of Points: " << cloud->size() << std::endl;
    std::cout << "Size of the voxel grid: " << 2 * radius << std::endl; // Multiplied with two as half size is used (corresponding to the radius)
    std::cout << "Number of Voxels: " << num_voxels << std::endl;
    std::cout << "Smoothing Kernel: " << smoothing_kernel_width << std::endl;



    // Specify the parameters of the algorithm
    const int grid_size = num_voxels * num_voxels * num_voxels;
    float voxel_step_size = (2 * radius) / num_voxels;
    float lrf_radius = sqrt(3)*radius; // Such that the circumscribed sphere is obtained

    // Initialize the voxel grid
    flann::Matrix<float> voxel_coordinates = initializeGridMatrix(num_voxels, voxel_step_size, voxel_step_size, voxel_step_size);

    // Compute the local reference frame for all the points
    float smoothing_factor = smoothing_kernel_width * (radius / num_voxels); // Equals half a voxel size so that 3X is 1.5 voxel

    // Check if all the points should be evaluated or only selected  ones
    std::string flag_all_points = "0";
    std::vector<int> evaluation_points;

    // Erase /r at the end of the filename (needed in linux environment)
    interest_points_file.erase(std::remove(interest_points_file.begin(), interest_points_file.end(), '\r'), interest_points_file.end());

    // If the keypoint file is not given initialize the ecaluation points with all the points in the point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr interest_points(new pcl::PointCloud<pcl::PointXYZ>);
    bool has_pc_interest_points = false;
    if (!interest_points_file.compare(flag_all_points))
    {
        std::vector<int> ep_temp(cloud->width);
        std::iota(ep_temp.begin(), ep_temp.end(), 0);
        std::iota(ep_temp.begin(), ep_temp.end(), 0);
        evaluation_points = ep_temp;
        ep_temp.clear();
    }
    else if (interest_points_file.substr(interest_points_file.length() - 4, 4).compare(".pcd") == 0) {
        if (pcl::io::loadPCDFile(interest_points_file, *interest_points) == -1) {
            std::cerr << "Interest points could not be loaded" << std::endl;
            return -1;
        }
        has_pc_interest_points = true;
    }
    else
    {
        if (fileExist(interest_points_file))
        {
            std::vector<int> ep_temp = readKeypoints(interest_points_file);
            evaluation_points = ep_temp;
            ep_temp.clear();
        }
        else
        {
            std::cout << "Keypoint file does not exsist or cannot be opened!!" << std::endl;
            return 1;
        }
    }

    if (has_pc_interest_points)
        std::cout << "Number of keypoints:" << interest_points->size() << "\n" << std::endl;
    else
        std::cout << "Number of keypoints:" << evaluation_points.size() << "\n" << std::endl;


    // Initialize the variables for the NN search and LRF computation
    std::vector<int> indices(cloud->width);
    std::vector<LRF> cloud_lrf(cloud->width);
    std::iota(indices.begin(), indices.end(), 0);
    std::vector <std::vector <int>> nearest_neighbors(cloud->width);
    std::vector <std::vector <int>> nearest_neighbors_smoothing(cloud->width);
    std::vector <std::vector <float>> nearest_neighbors_smoothing_dist(cloud->width);

    // Compute the local reference frame for the interes points (code adopted from https://www.researchgate.net/publication/310815969_TOLDI_An_effective_and_robust_approach_for_3D_local_shape_description
    // and not optimized)
    std::__success_type<std::chrono::nanoseconds>::type lrf_span;
    if (has_pc_interest_points) {
        auto t1_lrf = std::chrono::high_resolution_clock::now();
        toldiComputeLRF(cloud, interest_points, lrf_radius, 3 * smoothing_factor, cloud_lrf, nearest_neighbors, nearest_neighbors_smoothing, nearest_neighbors_smoothing_dist);
        auto t2_lrf = std::chrono::high_resolution_clock::now();
        lrf_span = t2_lrf - t1_lrf;
    }
    else {
        auto t1_lrf = std::chrono::high_resolution_clock::now();
        toldiComputeLRF(cloud, evaluation_points, lrf_radius, 3 * smoothing_factor, cloud_lrf, nearest_neighbors, nearest_neighbors_smoothing, nearest_neighbors_smoothing_dist);
        auto t2_lrf = std::chrono::high_resolution_clock::now();
        lrf_span = t2_lrf - t1_lrf;
    }

    // Compute the SDV representation for all the points
    std::string save_dir;
    if (output_file.compare("") == 0) {
        std::size_t found = data_file.find_last_of("/");
        std::string temp_token = data_file.substr(found + 1);
        std::size_t found2 = data_file.find_last_of(".");

        std::string save_file = temp_token.substr(0, found2);
        save_dir = output_folder + save_file;
    }
    else {
        save_dir = output_folder;
    }

    // Start the actual computation
    std::__success_type<std::chrono::nanoseconds>::type sdv_span;
    if (has_pc_interest_points) {
        auto t1 = std::chrono::high_resolution_clock::now();
        computeLocalDepthFeature(cloud, interest_points, nearest_neighbors, cloud_lrf, radius, voxel_coordinates, num_voxels, smoothing_factor, save_dir, output_file, chunk_size);
        auto t2 = std::chrono::high_resolution_clock::now();
        sdv_span = t2 - t1;
    }
    else {
        auto t1 = std::chrono::high_resolution_clock::now();
        computeLocalDepthFeature(cloud, evaluation_points, nearest_neighbors, cloud_lrf, radius, voxel_coordinates, num_voxels, smoothing_factor, save_dir, output_file, chunk_size);
        auto t2 = std::chrono::high_resolution_clock::now();
        sdv_span = t2 - t1;
    }
    std::cout << "\n---------------------------------------------------------" << std::endl;
    std::cout << "LRF computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(lrf_span).count()
              << " miliseconds\n";
    std::cout << "SDV computation took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(sdv_span).count()
              << " miliseconds\n";
    std::cout << "---------------------------------------------------------" << std::endl;


    return 0;

}
