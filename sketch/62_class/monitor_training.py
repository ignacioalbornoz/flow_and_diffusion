#!/usr/bin/env python3

import psutil
import time
import json
import os
import sys
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import queue

class TrainingMonitor:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.monitoring = True
        self.metrics = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_used_gb': [],
            'gpu_memory_used_gb': [],
            'gpu_memory_percent': [],
            'gpu_utilization': [],
            'disk_io_read_mb': [],
            'disk_io_write_mb': [],
            'network_sent_mb': [],
            'network_recv_mb': []
        }
        self.start_time = time.time()
        self.last_disk_io = psutil.disk_io_counters()
        self.last_network_io = psutil.net_io_counters()
        
        # Create monitoring directory
        self.monitor_dir = f"experiments/{experiment_id}/monitoring"
        os.makedirs(self.monitor_dir, exist_ok=True)
        
        # Initialize GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.device('cuda:1')  # Use GPU 1 as in training script
            print(f"GPU monitoring enabled for device: {self.device}")
        else:
            print("GPU not available, monitoring CPU only")
    
    def get_gpu_metrics(self):
        """Get GPU metrics if available"""
        if not self.gpu_available:
            return 0, 0, 0
        
        try:
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
            gpu_memory_used = torch.cuda.memory_reserved(self.device)
            gpu_memory_percent = (gpu_memory_used / gpu_memory) * 100
            gpu_memory_used_gb = gpu_memory_used / (1024**3)
            
            # Get GPU utilization (this is approximate)
            gpu_utilization = 0  # PyTorch doesn't provide direct GPU utilization
            # We can estimate based on memory usage patterns
            
            return gpu_memory_used_gb, gpu_memory_percent, gpu_utilization
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
            return 0, 0, 0
    
    def collect_metrics(self):
        """Collect system metrics"""
        timestamp = time.time()
        
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        
        # GPU metrics
        gpu_memory_used_gb, gpu_memory_percent, gpu_utilization = self.get_gpu_metrics()
        
        # Disk I/O
        current_disk_io = psutil.disk_io_counters()
        disk_read_mb = (current_disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024**2)
        disk_write_mb = (current_disk_io.write_bytes - self.last_disk_io.write_bytes) / (1024**2)
        self.last_disk_io = current_disk_io
        
        # Network I/O
        current_network_io = psutil.net_io_counters()
        network_sent_mb = (current_network_io.bytes_sent - self.last_network_io.bytes_sent) / (1024**2)
        network_recv_mb = (current_network_io.bytes_recv - self.last_network_io.bytes_recv) / (1024**2)
        self.last_network_io = current_network_io
        
        # Store metrics
        self.metrics['timestamps'].append(timestamp)
        self.metrics['cpu_percent'].append(cpu_percent)
        self.metrics['memory_percent'].append(memory_percent)
        self.metrics['memory_used_gb'].append(memory_used_gb)
        self.metrics['gpu_memory_used_gb'].append(gpu_memory_used_gb)
        self.metrics['gpu_memory_percent'].append(gpu_memory_percent)
        self.metrics['gpu_utilization'].append(gpu_utilization)
        self.metrics['disk_io_read_mb'].append(disk_read_mb)
        self.metrics['disk_io_write_mb'].append(disk_write_mb)
        self.metrics['network_sent_mb'].append(network_sent_mb)
        self.metrics['network_recv_mb'].append(network_recv_mb)
        
        return {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_used_gb': memory_used_gb,
            'gpu_memory_used_gb': gpu_memory_used_gb,
            'gpu_memory_percent': gpu_memory_percent,
            'gpu_utilization': gpu_utilization,
            'disk_io_read_mb': disk_read_mb,
            'disk_io_write_mb': disk_write_mb,
            'network_sent_mb': network_sent_mb,
            'network_recv_mb': network_recv_mb
        }
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_file = os.path.join(self.monitor_dir, "system_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def plot_metrics(self):
        """Create plots of system metrics"""
        if len(self.metrics['timestamps']) < 2:
            return
        
        # Convert timestamps to relative time in minutes
        start_time = self.metrics['timestamps'][0]
        time_minutes = [(t - start_time) / 60 for t in self.metrics['timestamps']]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'System Metrics - Experiment {self.experiment_id}', fontsize=16)
        
        # CPU Usage
        axes[0, 0].plot(time_minutes, self.metrics['cpu_percent'], 'b-', linewidth=2)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('CPU %')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Memory Usage
        axes[0, 1].plot(time_minutes, self.metrics['memory_percent'], 'r-', linewidth=2)
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Memory %')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Memory Used (GB)
        axes[0, 2].plot(time_minutes, self.metrics['memory_used_gb'], 'g-', linewidth=2)
        axes[0, 2].set_title('Memory Used (GB)')
        axes[0, 2].set_xlabel('Time (minutes)')
        axes[0, 2].set_ylabel('Memory (GB)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # GPU Memory
        if self.gpu_available:
            axes[1, 0].plot(time_minutes, self.metrics['gpu_memory_used_gb'], 'purple', linewidth=2)
            axes[1, 0].set_title('GPU Memory Used (GB)')
            axes[1, 0].set_xlabel('Time (minutes)')
            axes[1, 0].set_ylabel('GPU Memory (GB)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'GPU Not Available', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('GPU Memory')
        
        # Disk I/O
        axes[1, 1].plot(time_minutes, self.metrics['disk_io_read_mb'], 'orange', linewidth=2, label='Read')
        axes[1, 1].plot(time_minutes, self.metrics['disk_io_write_mb'], 'red', linewidth=2, label='Write')
        axes[1, 1].set_title('Disk I/O (MB/s)')
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('MB/s')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Network I/O
        axes[1, 2].plot(time_minutes, self.metrics['network_sent_mb'], 'blue', linewidth=2, label='Sent')
        axes[1, 2].plot(time_minutes, self.metrics['network_recv_mb'], 'green', linewidth=2, label='Received')
        axes[1, 2].set_title('Network I/O (MB/s)')
        axes[1, 2].set_xlabel('Time (minutes)')
        axes[1, 2].set_ylabel('MB/s')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.monitor_dir, "system_metrics_plot.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"System metrics plot saved: {plot_file}")
    
    def log_current_metrics(self):
        """Log current system metrics to file"""
        if not self.metrics['timestamps']:
            return
        
        latest = {
            'cpu_percent': self.metrics['cpu_percent'][-1],
            'memory_percent': self.metrics['memory_percent'][-1],
            'memory_used_gb': self.metrics['memory_used_gb'][-1],
            'gpu_memory_used_gb': self.metrics['gpu_memory_used_gb'][-1],
            'gpu_memory_percent': self.metrics['gpu_memory_percent'][-1]
        }
        
        elapsed_time = time.time() - self.start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        # Log to file instead of printing
        log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] System Metrics ({hours:02d}:{minutes:02d} elapsed) - CPU: {latest['cpu_percent']:.1f}%, Memory: {latest['memory_percent']:.1f}% ({latest['memory_used_gb']:.1f} GB)"
        if self.gpu_available:
            log_entry += f", GPU Memory: {latest['gpu_memory_percent']:.1f}% ({latest['gpu_memory_used_gb']:.1f} GB)"
        
        # Write to log file
        with open(os.path.join(self.monitor_dir, "monitor.log"), "a") as f:
            f.write(log_entry + "\n")
    
    def monitor_loop(self):
        """Main monitoring loop"""
        # Log startup message
        with open(os.path.join(self.monitor_dir, "monitor.log"), "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting system monitoring for experiment: {self.experiment_id}\n")
        
        try:
            while self.monitoring:
                metrics = self.collect_metrics()
                self.log_current_metrics()
                
                # Save metrics every 10 samples
                if len(self.metrics['timestamps']) % 10 == 0:
                    self.save_metrics()
                
                # Create plot every 50 samples
                if len(self.metrics['timestamps']) % 50 == 0:
                    self.plot_metrics()
                
                time.sleep(10)  # Collect metrics every 10 seconds
                
        except KeyboardInterrupt:
            with open(os.path.join(self.monitor_dir, "monitor.log"), "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Stopping monitoring...\n")
            self.monitoring = False
        
        # Final save and plot
        self.save_metrics()
        self.plot_metrics()
        with open(os.path.join(self.monitor_dir, "monitor.log"), "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Final metrics saved to: {self.monitor_dir}\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python monitor_training.py <experiment_id>")
        sys.exit(1)
    
    experiment_id = sys.argv[1]
    monitor = TrainingMonitor(experiment_id)
    monitor.monitor_loop()

if __name__ == "__main__":
    main()
