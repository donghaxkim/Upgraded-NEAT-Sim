import pygame
import neat
import os
import random
import numpy as np
from agent import Agent
from environment import Environment
from visualization import Visualizer
from config import *

def create_agents(config, pop_size):
    """Create a population of agents"""
    agents = []
    # Create prey
    for _ in range(pop_size):
        x = random.randint(AGENT_RADIUS, WINDOW_WIDTH - AGENT_RADIUS)
        y = random.randint(AGENT_RADIUS, WINDOW_HEIGHT - AGENT_RADIUS)
        genome = neat.DefaultGenome(0)
        genome.configure_new(config.genome_config)
        agents.append(Agent(x, y, genome, config, is_predator=False))
    
    # Create predators
    for _ in range(PREDATOR_COUNT):
        x = random.randint(PREDATOR_RADIUS, WINDOW_WIDTH - PREDATOR_RADIUS)
        y = random.randint(PREDATOR_RADIUS, WINDOW_HEIGHT - PREDATOR_RADIUS)
        genome = neat.DefaultGenome(0)
        genome.configure_new(config.genome_config)
        agents.append(Agent(x, y, genome, config, is_predator=True))
    
    return agents

def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Neural Network Evolution")
    clock = pygame.time.Clock()

    # Initialize NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config.txt')
    
    # Create initial population
    agents = create_agents(config, config.pop_size)
    environment = Environment()
    visualizer = Visualizer(screen)

    # Main simulation loop
    running = True
    paused = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    agents = create_agents(config, config.pop_size)
                    environment.reset()
                    visualizer = Visualizer(screen)
                elif event.key == pygame.K_q:
                    running = False

        if not paused:
            # Update environment and agents
            environment.update(agents)
            for agent in agents:
                agent.update(environment.foods, agents)

            # Handle predator-prey interactions
            for predator in [a for a in agents if a.is_predator and a.alive]:
                for prey in [a for a in agents if not a.is_predator and a.alive]:
                    dist = np.sqrt((predator.x - prey.x)**2 + (predator.y - prey.y)**2)
                    if dist < predator.radius + prey.radius:
                        prey.alive = False
                        predator.fitness += 5
                        predator.energy += FOOD_ENERGY

            # Check if all prey are dead
            if all(not agent.alive for agent in agents if not agent.is_predator):
                # Create next generation
                visualizer.increment_generation()
                
                # Separate predators and prey
                prey_genomes = [(a.genome, a.fitness) for a in agents if not a.is_predator]
                pred_genomes = [(a.genome, a.fitness) for a in agents if a.is_predator]
                
                prey_genomes.sort(key=lambda x: x[1], reverse=True)
                pred_genomes.sort(key=lambda x: x[1], reverse=True)
                
                # Create new population
                new_agents = []
                
                # New prey
                for i in range(config.pop_size):
                    parent = random.choice(prey_genomes[:config.pop_size//2])[0]
                    child = neat.DefaultGenome(i)
                    child.configure_crossover(parent, parent, config.genome_config)
                    child.mutate(config.genome_config)
                    x = random.randint(AGENT_RADIUS, WINDOW_WIDTH - AGENT_RADIUS)
                    y = random.randint(AGENT_RADIUS, WINDOW_HEIGHT - AGENT_RADIUS)
                    new_agents.append(Agent(x, y, child, config, is_predator=False))
                
                # New predators
                for i in range(PREDATOR_COUNT):
                    parent = random.choice(pred_genomes[:PREDATOR_COUNT//2])[0]
                    child = neat.DefaultGenome(i + config.pop_size)
                    child.configure_crossover(parent, parent, config.genome_config)
                    child.mutate(config.genome_config)
                    x = random.randint(PREDATOR_RADIUS, WINDOW_WIDTH - PREDATOR_RADIUS)
                    y = random.randint(PREDATOR_RADIUS, WINDOW_HEIGHT - PREDATOR_RADIUS)
                    new_agents.append(Agent(x, y, child, config, is_predator=True))
                
                agents = new_agents
                environment.reset()

        # Draw everything
        screen.fill((0, 0, 0))
        environment.draw(screen)
        for agent in agents:
            agent.draw(screen)
        
        # Draw statistics and neural network visualization
        visualizer.draw_stats(agents)
        if agents:
            best_agent = max(agents, key=lambda x: x.fitness)
            visualizer.draw_neural_network(best_agent, 
                                        WINDOW_WIDTH - 200, 50, 150, 100)

        pygame.display.flip()
        clock.tick(FPS)

    # Save fitness history plot
    visualizer.plot_fitness_history()
    pygame.quit()

if __name__ == "__main__":
    main()