/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  if (is_initialized)
  {
    return;
  }
  
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;
  particles.resize(num_particles);
  weights.resize(num_particles);
  
  for (int i = 0; i < num_particles; i++)
  {    
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    Particle part;
    part.id = i;
    part.x = dist_x(gen);
    part.y = dist_y(gen);
    part.theta = dist_theta(gen);
    part.weight = 1.0;
    
    
    particles[i] = part;
    weights[i] = 1.0;
  }
  is_initialized=true;
    
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double std_x, std_y, std_theta;
  std_x = std_pos[0];
  std_y = std_pos[1];
  std_theta = std_pos[2];
  for (int i = 0; i < num_particles; i++)
  {
    double theta = particles[i].theta;
    if (fabs(yaw_rate) < 0.0001)
    {      
      //std::cout << "before x " << particles[i].x << " y " << particles[i].y << std::endl;      
      particles[i].x += velocity*delta_t*cos(theta);
      particles[i].y += velocity*delta_t*sin(theta);
      //std::cout << "after x " << particles[i].x << " y " << particles[i].y << std::endl;
    }
    else
    {
      //std::cout << "before x " << particles[i].x << " y " << particles[i].y << " theta " << particles[i].theta << std::endl;
      particles[i].x += velocity/yaw_rate * (sin(theta + yaw_rate*delta_t)-sin(theta));
      particles[i].y += velocity/yaw_rate * (cos(theta) - cos(theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate*delta_t;
      //std::cout << "after x " << particles[i].x << " y " << particles[i].y << " theta " << particles[i].theta << std::endl;
    }
                                              
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);
    std::default_random_engine gen;
    
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);  
  }
                                              
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.    
   */
  for (int i = 0; i < observations.size(); i++)
  { 
    //switched i and j index limits together to test out a theory found whilst debugging
    vector<double> distances;
    double pred_x, pred_y;
    int pred_id;
    
    int obs_id = observations[i].id;
    double min_dist=10000;
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;
    for (int j = 0; j < predicted.size(); j++)
    {       
      //std::cout << "observation " << i << " " << observations[i].x << " " << observations[i].y << std::endl;
      //std::cout << "predictions: " << predicted[j].x << " " << predicted[j].y << std::endl;
      pred_x = predicted[j].x;
      pred_y = predicted[j].y;
      pred_id = predicted[j].id;
      if (min_dist > dist(pred_x, pred_y, obs_x, obs_y))
      {
        min_dist = dist(pred_x, pred_y, obs_x, obs_y);
        obs_id = pred_id;
      }      
      //distances.push_back(min_dist);            
      //std::cout << "ID " << obs_id << " distance " << distances[j] << " x " << obs_x << " y " << obs_y << std::endl; 
    }
    //double min_idx = min_element(distances.begin(), distances.end()) - distances.begin();
    //only setting the ID 
    observations[i].id = obs_id;
    //std::cout << observations[obs_id].id << std::endl;
    //associations.push_back(predicted[min_idx].id);
    //sense_x.push_back(pred_x);
    //sense_y.push_back(pred_y);
    //std::cout << "observation " << i << " " << observations[i].x << " " << observations[i].y << std::endl;
    //std::cout << "predictions: " << predicted[min_idx].x << " " << predicted[min_idx].y << std::endl;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
     
  //vector<double> x_map_arr;
  //vector<double> y_map_arr;
  vector<double> p_weights;
  double sum_particle_weight = 0.0;
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  //weights.clear();
  for (int i = 0; i < num_particles; i++)
  {
    
    double particle_x, particle_y, particle_theta;
    particle_x = particles[i].x;
    particle_y = particles[i].y;
    particle_theta = particles[i].theta;
    
    vector<LandmarkObs> predictions; //this will contain map landmark predictions
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      double landmark_x, landmark_y, landmark_id;
      landmark_x = map_landmarks.landmark_list[j].x_f;
      landmark_y = map_landmarks.landmark_list[j].y_f;
      landmark_id = map_landmarks.landmark_list[j].id_i;
      //std::cout << "Landmark " << j << " x, y, id" << landmark_x << ", " << landmark_y << ", " << landmark_id << std::endl;
      if (dist(landmark_x, landmark_y, particle_x, particle_y) <= sensor_range)
      {
        //if landmark falls within particle's position and sensor range, add to prediction list
        predictions.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }                  
    }
    
    vector<LandmarkObs> map_observations;
    for (int j = 0; j < observations.size(); j++)
	{
      double obs_x = observations[j].x;
      double obs_y = observations[j].y;
      double obs_id = observations[j].id;
      double x_map = particle_x + (cos(particle_theta) * obs_x) - (sin(particle_theta) * obs_y);
      double y_map = particle_y + (sin(particle_theta) * obs_x) + (cos(particle_theta) * obs_y);
      map_observations.push_back(LandmarkObs{obs_id, x_map, y_map});
      //LandmarkObs mops;
      //mops.x = x_map;
      //mops.y = y_map;
      //mops.id = obs_id;
      //map_observations.push_back(mops);
      //std::cout << "x, y, id " << x_map << ", " << y_map << ", " << observations[j].id << std::endl;
    }
    
    
    //calling the dataAssociation function
    dataAssociation(predictions, map_observations);
    
    weights[i] = 1.0;
    particles[i].weight = 1.0;  
    for (int j = 0; j < map_observations.size(); j++)
    {     
        //std::cout << "obs x " << map_observations[j].x << " y " << map_observations[j].y << " id " << map_observations[j].id << 							std::endl;
      double obs_x = map_observations[j].x;
      double obs_y = map_observations[j].y;
      double obs_id = map_observations[j].id;
      double pred_x;
      double pred_y;
      double pred_id;
      int k = 0;
      bool found = false;
      while (!found && k < predictions.size())
      {
        pred_id = predictions[k].id;
        if (pred_id == obs_id)
        {
          found = true;
          pred_x = predictions[k].x;
          pred_y = predictions[k].y;
          //std::cout << " pred x " << predictions[k].x << " y " << predictions[k].y << " id " << predictions[k].id << std::endl;
          //according to debugging, we find out that prediction id is never matched with observation
          //moved pred_id to before if statement          
        }         
        k++;
      }
      
      double exponent;
      double gauss_norm = 1 / (2 * M_PI * std_x * std_x);
      //std::cout << "map obs " << map_observations[j].x << " " << map_observations[j].y << std::endl;
      //std::cout << "pred " << pred_x << " " << pred_y << std::endl;
      double dx = obs_x - pred_x;
      double dy = obs_y - pred_y;
      exponent = (pow(dx, 2) / (2 * pow(std_x, 2))) + (pow(dy, 2) / (2 * pow(std_y, 2)));  
      double weight = gauss_norm * exp(-exponent);
      //std::cout << "exponent " << exponent << " gauss norm " << gauss_norm << std::endl; 
      if (weight == 0)
      {
        particles[i].weight *= 0.00001;
      }
      else
      {
        particles[i].weight *= weight; 
      }
      //std::cout << weight << " weight" << std::endl;
      //std::cout << weights[i] << " weight" << std::endl;
    }
    
    //p_weights.push_back(particles[i].weight);
    //weights.push_back(particles[i].weight);
      
    weights[i] = particles[i].weight;
    //std::cout << "sum particle weight " << sum_particle_weight << std::endl;
    //particles[i].weight = weight;
    sum_particle_weight += particles[i].weight;
    //weights[i] = weight;
  }
  
  /**
  if (sum_particle_weight != 0)
  {
    for (int i = 0; i < num_particles; i++)
    {
      particles[i].weight /= sum_particle_weight;
      weights[i] = particles[i].weight;
    }
    
  }
  */
  /**
  
  */
  //weights = p_weights; 
  //std::cout << "weights size " << weights.size() << std::endl;
  
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<double> new_weights;
  vector<double> prev_weights;
  prev_weights.reserve(num_particles);
  std::default_random_engine gen;
  vector<Particle> particles_new;
  particles_new.reserve(num_particles);
    
  
  for (int i = 0; i < num_particles; i++)
  {
    prev_weights.push_back(particles[i].weight);    
  }
  
  /**
  std::discrete_distribution<> dd(weights.begin(), weights.end());
  while (particles_new.size() < weights.size())
  {
    int id = dd(gen);
    particles_new.push_back(particles[id]);
    new_weights.push_back(particles[id].weight);
  }
  particles = particles_new;
  weights = new_weights;
  */
  
  
  double max_weight = *max_element(prev_weights.begin(), prev_weights.end());
  //double min_weight = *min_element(prev_weights.begin(), prev_weights.end());
  
  uniform_int_distribution<int> idx(0, num_particles - 1);
  int index = idx(gen);
  double beta = 0.0;
  
  uniform_real_distribution<double> bt(0.0, max_weight);
  for (int i = 0; i < num_particles; i++)
  {
    beta += bt(gen)*2.0;
    while (beta > prev_weights[index])
    {
      beta -= prev_weights[index];
      index = (index + 1) % num_particles;
    }
    //std::cout << "index " << index << std::endl;
    particles_new.push_back(particles[index]);
    new_weights.push_back(particles[index].weight);
  }
  
  particles = particles_new;
  //weights = new_weights;
  //weights.clear();
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}