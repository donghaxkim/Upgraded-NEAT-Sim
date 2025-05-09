import numpy as np
import pygame
import neat
from config import *
from collections import deque

class Agent:
    def __init__(self, x, y, genome, config, is_predator=False):
        self.x = x
        self.y = y
        self.radius = PREDATOR_RADIUS if is_predator else AGENT_RADIUS
        self.energy = PREDATOR_ENERGY if is_predator else AGENT_ENERGY
        self.speed = PREDATOR_SPEED if is_predator else AGENT_SPEED
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.fitness = 0
        self.alive = True
        self.angle = 0
        self.is_predator = is_predator
        self.memory = deque(maxlen=MEMORY_SIZE)
        for _ in range(MEMORY_SIZE):
            self.memory.append(0)  # Initialize memory with zeros

    def get_vision_inputs(self, foods, agents):
        """Get inputs from vision cone"""
        vision_inputs = []
        
        # Check objects in vision cone
        for angle in np.linspace(-VISION_ANGLE/2, VISION_ANGLE/2, 5):
            ray_angle = self.angle + np.radians(angle)
            ray_x = np.cos(ray_angle)
            ray_y = np.sin(ray_angle)
            
            # Cast ray
            closest_dist = VISION_RANGE
            for food in foods:
                if food.eaten:
                    continue
                    
                dx = food.x - self.x
                dy = food.y - self.y
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < closest_dist:
                    # Check if in vision cone
                    angle_to_food = np.arctan2(dy, dx)
                    angle_diff = abs(angle_to_food - ray_angle)
                    if angle_diff < np.radians(VISION_ANGLE/2):
                        closest_dist = dist
            
            vision_inputs.append(closest_dist / VISION_RANGE)
        
        return vision_inputs

    def get_inputs(self, foods, agents):
        """Get neural network inputs based on environment state"""
        inputs = []
        
        # Find nearest food and predator/prey
        nearest_food = None
        nearest_agent = None
        min_food_dist = float('inf')
        min_agent_dist = float('inf')
        
        for food in foods:
            if food.eaten:
                continue
            dist = np.sqrt((food.x - self.x)**2 + (food.y - self.y)**2)
            if dist < min_food_dist:
                min_food_dist = dist
                nearest_food = food
                
        for agent in agents:
            if agent == self or not agent.alive:
                continue
            if agent.is_predator != self.is_predator:  # Only interested in opposite type
                dist = np.sqrt((agent.x - self.x)**2 + (agent.y - self.y)**2)
                if dist < min_agent_dist:
                    min_agent_dist = dist
                    nearest_agent = agent

        # Basic inputs
        if nearest_food:
            inputs.append(min_food_dist / np.sqrt(WINDOW_WIDTH**2 + WINDOW_HEIGHT**2))
            dx = nearest_food.x - self.x
            dy = nearest_food.y - self.y
            inputs.append(np.arctan2(dy, dx) / (2 * np.pi))
        else:
            inputs.extend([1.0, 0.0])

        # Predator/Prey inputs
        if nearest_agent:
            inputs.append(min_agent_dist / np.sqrt(WINDOW_WIDTH**2 + WINDOW_HEIGHT**2))
            dx = nearest_agent.x - self.x
            dy = nearest_agent.y - self.y
            inputs.append(np.arctan2(dy, dx) / (2 * np.pi))
        else:
            inputs.extend([1.0, 0.0])

        # Vision inputs
        inputs.extend(self.get_vision_inputs(foods, agents))
        
        # Memory inputs
        inputs.extend(list(self.memory))
        
        # Current state
        inputs.append(self.energy / (PREDATOR_ENERGY if self.is_predator else AGENT_ENERGY))
        inputs.append(self.speed / (PREDATOR_SPEED if self.is_predator else AGENT_SPEED))
        
        return inputs

    def update(self, foods, agents):
        """Update agent state based on neural network output"""
        if not self.alive:
            return

        # Get neural network inputs
        inputs = self.get_inputs(foods, agents)
        
        # Get neural network output
        output = self.net.activate(inputs)
        
        # Update memory with current action
        self.memory.append(output[0])  # Store turning decision
        
        # Update movement
        self.angle += (output[0] - 0.5) * np.pi
        self.speed = output[1] * (PREDATOR_SPEED if self.is_predator else AGENT_SPEED)
        
        # Update position
        self.x += np.cos(self.angle) * self.speed
        self.y += np.sin(self.angle) * self.speed
        
        # Keep agent within bounds
        self.x = np.clip(self.x, self.radius, WINDOW_WIDTH - self.radius)
        self.y = np.clip(self.y, self.radius, WINDOW_HEIGHT - self.radius)
        
        # Decrease energy
        self.energy -= ENERGY_DECAY
        
        # Check if agent is dead
        if self.energy <= 0:
            self.alive = False

    def draw(self, screen):
        """Draw the agent on the screen"""
        if not self.alive:
            return

        # Draw agent body
        color = (255, 0, 0) if self.is_predator else (0, 255, 0)
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
        
        # Draw direction indicator
        end_x = self.x + np.cos(self.angle) * self.radius
        end_y = self.y + np.sin(self.angle) * self.radius
        pygame.draw.line(screen, (255, 255, 255), (self.x, self.y), (end_x, end_y), 2)
        
        # Draw vision cone
        if not self.is_predator:  # Only draw vision cone for prey
            left_angle = self.angle - np.radians(VISION_ANGLE/2)
            right_angle = self.angle + np.radians(VISION_ANGLE/2)
            
            points = [(self.x, self.y),
                     (self.x + np.cos(left_angle) * VISION_RANGE,
                      self.y + np.sin(left_angle) * VISION_RANGE),
                     (self.x + np.cos(right_angle) * VISION_RANGE,
                      self.y + np.sin(right_angle) * VISION_RANGE)]
            
            pygame.draw.polygon(screen, (100, 100, 100), points, 1)

    def eat(self, food):
        """Consume food and gain energy"""
        max_energy = PREDATOR_ENERGY if self.is_predator else AGENT_ENERGY
        self.energy = min(max_energy, self.energy + FOOD_ENERGY)
        self.fitness += 1