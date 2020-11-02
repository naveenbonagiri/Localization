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

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
	/**
	* TODO: Set the number of particles. Initialize all particles to 
	*   first position (based on estimates of x, y, theta and their uncertainties
	*   from GPS) and all weights to 1. 
	* TODO: Add random Gaussian noise to each particle.
	* NOTE: Consult particle_filter.h for more information about this method 
	*   (and others in this file).
	*/
	num_particles = 200;  // TODO: Set the number of particles

	/* create random number generator class object type */
	std::default_random_engine rand_num_gen;

	/* standard deviation variables */
	double std_x{}, std_y{}, std_theta{};

	/* create object for particle */
	Particle particle_obj;

	/* create Gaussian distribution for noise x, y and theta around mean 0 and standard deviation */
	std::normal_distribution<double> noise_x(0, std_x);
	std::normal_distribution<double> noise_y(0, std_y);
	std::normal_distribution<double> noise_theta(0, std_theta);
     
    /* Get standard deviation values for x, y and theta */
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    /* create Gaussian distribution for distance x, y and theta around mean 0 and standard deviation */
    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_theta);
	
	/* allocate memmory for defined particles */
	particles.reserve(num_particles);

    /* intialize particle object */
    for (int i = 0; i < num_particles; i++) 
	{
        particle_obj.id = i;
        particle_obj.x = dist_x(rand_num_gen) + noise_x(rand_num_gen);
        particle_obj.y = dist_y(rand_num_gen) + noise_y(rand_num_gen);
        particle_obj.theta = dist_theta(rand_num_gen) + noise_theta(rand_num_gen);
        particle_obj.weight = 1.0;
        particles.push_back(particle_obj);
    }

    /* set the flag to indicate initialization is complete */
    is_initialized = true;	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) 
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   
    /* create random number generator class object type */
    std::default_random_engine RandNumGen;
   
    /* standard deviation variables */
    double std_x{}, std_y{}, std_theta{};

    /* Get standard deviation values for x, y and theta */
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

	/* create Gaussian distribution for distance x, y and theta around mean 0 and standard deviation */
    std::normal_distribution<double> dist_x(0, std_x);
    std::normal_distribution<double> dist_y(0, std_y);
    std::normal_distribution<double> dist_theta(0, std_theta);

    /* update each particle after delta t */
    for_each(particles.begin(), particles.end(), [&](Particle &particle)
	{
		if (fabs(yaw_rate) < 0.001) 
		{
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		}
		else 
		{
			particle.x += velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
			particle.y += velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
			particle.theta += yaw_rate * delta_t;
		}

		/* update particle  */
		particle.x += dist_x(RandNumGen);
		particle.y += dist_y(RandNumGen);
		particle.theta += dist_theta(RandNumGen);
	});
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
    
    double nearest_neighbour{}, min_dist{};
    int landmark_id{};

    /* identify association for all sensor measurements */
    for (auto& obs_meas : observations) 
	{
        /* initialize minimum distance with maximum value */
        min_dist = std::numeric_limits<double>::max();

        /* set particle id to default value */
        obs_meas.id = -1;

        /* loop through all the land marks */
        for (const auto& pred_meas : predicted) 
		{
            /* identify nearest neighbour using predicted and observed information */
            nearest_neighbour = dist(pred_meas.x, pred_meas.y, obs_meas.x, obs_meas.y);

            /* determine minimum distance */
            if (nearest_neighbour < min_dist) 
			{
                min_dist = nearest_neighbour;
                landmark_id = pred_meas.id;
            }
        }

        /* update observed measurement id */
        obs_meas.id = landmark_id;
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
     /* repeat the steps for each particle */
    for (auto& particle : particles) 
	{
        /* vector for transormed coordinates */
        std::vector<LandmarkObs> TransformedCoordinates{};
		
        /* vector for predicated coordinates */
        std::vector<LandmarkObs> PredictionCoordinates{};

        /* transform sensor observation landmarks to map coordinate system */
        TransformedCoordinates = TransformCarMeasToMapCoordinates(observations, particle);

        /* Assoicate transformed observations to landmarks on the map using predicated measurements */
        PredictionCoordinates = GetPredicatedCoordinates(map_landmarks, particle, sensor_range);
        dataAssociation(PredictionCoordinates, TransformedCoordinates);

        /* update particle weight */
        UpdateParticleWeight(TransformedCoordinates, PredictionCoordinates, particle, std_landmark);
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
     std::vector<Particle> new_particles(num_particles);      /* vector for new particles */
     std::default_random_engine RandomNumGen; 	 /* create random number generator class object type */
     std::vector<double> weights; 	 /* vector for weights */

     /* Update weights */
     for (unsigned int idx = 0; idx < particles.size(); idx++) 
	 {
         weights.push_back(particles[idx].weight);
     }

     /* discrete distribution of weights */
     std::discrete_distribution<size_t> DiscDistrIndex(weights.begin(), weights.end());

     /* based on the weight, create new particle set */
     for (unsigned int i = 0; i < particles.size(); i++) 
	 {
         NewParticlesSet[i] = particles[DiscDistrIndex(RandomNumGen)];
     }

     /* update into original particle vector */
     particles = std::move(NewParticlesSet);
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
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) 
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) 
{
  vector<double> v;

  if (coord == "X") 
  {
    v = best.sense_x;
  } 
  else 
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

std::vector<LandmarkObs> ParticleFilter::TransformCarMeasToMapCoordinates(const vector<LandmarkObs>& observations, Particle &particle) {

    /* Initialize particle x, y co-ordinates, and its sine and cos theta */
    double x_p{}, y_p{}, sin_theta{}, cos_theta{};
    /* Initialize vector to store the transormed co-ordinates */
    std::vector<LandmarkObs> TransformedCoordinates{};
    /* Initialize the observed measurements for every particle */
    double x_c{}, y_c{}, obs_id{};

    /* Set the size for the transformed co-ordinates */
    TransformedCoordinates.reserve(observations.size());

    /* Get the x and y co-ordinates of the particle */
    x_p = particle.x;
    y_p = particle.y;

    /* Get the sine and cos of the particle */
    sin_theta = sin(particle.theta);
    cos_theta = cos(particle.theta);    

    /* Store the transformed co-ordinates */
    std::transform(observations.begin(), observations.end(), std::back_inserter(TransformedCoordinates),
        [&](LandmarkObs obs_meas) {
            /* Get the x and y co-ordinates of the observed measurements */
            x_c = obs_meas.x;
            y_c = obs_meas.y;
            obs_id = obs_meas.id;

            /* Intialize the object of the structure LandmarkObs to store values */
            LandmarkObs trans_cord{};

            /* Update the structure */
            trans_cord.x = x_p + cos_theta * x_c - sin_theta * y_c;
            trans_cord.y = y_p + sin_theta * x_c + cos_theta * y_c;
            trans_cord.id = obs_id;

            return trans_cord;
        });

    return TransformedCoordinates;
}

std::vector<LandmarkObs> ParticleFilter::GetPredicatedCoordinates(const Map& map_landmarks, Particle& particle, double sensor_range) {

    /* Initialize vector to store the global co-ordinates */
    std::vector<LandmarkObs> PredictionCoordinates{};

    for (const auto& glob_cord : map_landmarks.landmark_list) {
        /* Define structure for landmark */
        LandmarkObs map{};
        double x_p{}, y_p{};
        double distance{};

        /* Get the x and y co-ordinates of the particle */
        x_p = particle.x;
        y_p = particle.y;

        /* Check whether the distance between the particle and the map landmark is within the sensor range */
        distance = dist(x_p, y_p, glob_cord.x_f, glob_cord.y_f);

        /* Update landmark structure if distance is within the sensor range */
        if (distance < sensor_range) {

            map.x = glob_cord.x_f;
            map.y = glob_cord.y_f;
            map.id = glob_cord.id_i;

            /* Update the global co-ordinate structure */
            PredictionCoordinates.push_back(map);
        }        
    }

    return PredictionCoordinates;
}

void ParticleFilter::UpdateParticleWeight(const std::vector<LandmarkObs>& TransformedCoordinates, const std::vector<LandmarkObs>& PredictionCoordinates,
       Particle& particle, double std_landmark[]) {

    /* Initialize particle weight to 1 at every loop */
    particle.weight = 1;
    /* Set values for multi-variate Gaussian distribution */
    auto cov_x = std_landmark[0] * std_landmark[0];
    auto cov_y = std_landmark[1] * std_landmark[1];
    auto normalizer = 2.0 * M_PI * std_landmark[0] * std_landmark[1];

    /* Check if the transformed cordinates id gets matched with global cordinates id */
    for (unsigned int i = 0; i < TransformedCoordinates.size(); i++) {
        for (unsigned int j = 0; j < PredictionCoordinates.size(); j++) {
            if (TransformedCoordinates[i].id == PredictionCoordinates[j].id) {
                auto diff_x = TransformedCoordinates[i].x - PredictionCoordinates[j].x;
                auto diff_y = TransformedCoordinates[i].y - PredictionCoordinates[j].y;

                particle.weight *= exp(-(diff_x * diff_x / (2 * cov_x) + diff_y * diff_y / (2 * cov_y))) / normalizer;
            }
        }
    }
}
