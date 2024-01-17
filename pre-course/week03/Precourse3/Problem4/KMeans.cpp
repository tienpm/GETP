// KMeans.cpp
#include "KMeans.h"
#include "Point.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <algorithm>

// TODO: Fix the random seed to ensure consistent results across different program executions.
unsigned int seed = 42;

KMeans::KMeans(int num_points, int k) : k(k) {
    generateRandomDataPoints(num_points);
    initializeCentroids();
}

void KMeans::initializeCentroids() {
     // TODO: Implement this function
    // Initialize the centroids of clusters randomly.
    shuffle(data_points.begin(), data_points.end(), std::default_random_engine(seed));
    centroids.assign(data_points.begin(), data_points.begin() + k);
}

void KMeans::generateRandomDataPoints(int num_points) {
    // TODO: Implement this function
    // Generate a given number of random data points within a specified range ( 0< x, y<50)
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 50.0);
    data_points.reserve(num_points);
    for (int i = 0; i < num_points; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        data_points.emplace_back(x, y);
    }
}

std::vector<int> KMeans::assignToClusters() {
    // TODO: Implement this function
    // Assign each point to a cluster of the nearest centroid
    std::vector<int> cluster_labels;
    for (Point& point : data_points) {
        int closest_centroid_index = 0;
        double closest_dist = calculateDistance(point, centroids[0]);
        for (int i = 1; i < k; ++i) {
            double dist = calculateDistance(point, centroids[i]);
            if (dist < closest_dist) {
                closest_dist = dist;
                closest_centroid_index = i;
            }
        }
        cluster_labels.push_back(closest_centroid_index); // Assuming Point has a setCluster method
    }

    return cluster_labels;
}

void KMeans::updateCentroids(const std::vector<int> &labels) {
    // TODO: Implement this function
    // Update the centroids of clusters based on the current assignment of data points.
    std::vector<std::vector<Point>> clusters(k);
    for (size_t i = 0; i < labels.size(); i++) {
        clusters[labels[i]].push_back(data_points[i]); 
    }

    for (int i = 0; i < k; ++i) {
        Point& centroid = centroids[i];
        if (!clusters[i].empty()) {
            // Calculate the mean of the points in the cluster
            double sum_x = 0, sum_y = 0; // Assuming Point has x and y coordinates
            for (const Point& point : clusters[i]) {
                sum_x += point.x;
                sum_y += point.y;
            }
            centroid.x = sum_x / clusters[i].size();
            centroid.y = sum_y / clusters[i].size();
        }
    }
}

double KMeans::calculateDistance(const Point& p1, const Point& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

void KMeans::run(int max_iterations) {
    // TODO: Implement K-means algorithm and print the coordinates of each cluster centroid After the maximum number of iterations
    for (int i = 0; i < max_iterations; ++i) {
        std::vector<int> labels = assignToClusters();
        std::vector<Point> old_centroids = centroids;
        updateCentroids(labels);
        // if centoids no longer change significantly then stop
        bool stop = true;
        for (int i = 0; i < k; i++) {
            if (std::abs(old_centroids[i].x - centroids[i].x) > 1e-9 or 
                std::abs(old_centroids[i].y - centroids[i].y) > 1e-9) {
                stop = false;
                break;
            }
        }
        if (stop) break;
    }
    
    std::cout << "========== List Centroids ==========\n";
    for (size_t  i = 0; i < centroids.size(); i++) {
        std::cout << "Centroid " << i+1 << ": (" << centroids[i].x << "," << centroids[i].y << ")\n"; 
    }
    std::cout << "====================================\n";
}
