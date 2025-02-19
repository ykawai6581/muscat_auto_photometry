from IPython.display import clear_output

from tqdm import tqdm
from io import StringIO

class ProgressManager:
    def __init__(self, nccd):
        self.nccd = nccd
        self.progress_bars = {}
        self.main_bar = None
        self.start_line = 0
        
    def create_bars(self):
        # Store the current cursor position
        self.start_line = 0
        
        # Create main progress bar for CCDs
        self.main_bar = tqdm(
            total=self.nccd,
            desc="CCDs",
            position=self.start_line,
            leave=True,
            ncols=80,
            bar_format='{desc:<5} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
    def create_ccd_bar(self, ccd_num, total_files):
        # Create individual CCD progress bars
        position = ccd_num + 1
        self.progress_bars[ccd_num] = tqdm(
            total=total_files,
            desc=f"CCD {ccd_num}",
            position=position,
            leave=True,
            ncols=80,
            bar_format='{desc:<5} {bar:10} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        return self.progress_bars[ccd_num]
    
    def update_main(self):
        self.main_bar.update(1)
    
    def close_all(self):
        for bar in self.progress_bars.values():
            bar.close()
        self.main_bar.close()

class JupyterProgressManager(ProgressManager):
    def __init__(self, nccd):
        super().__init__(nccd)
        self.output = StringIO()
        
    def create_bars(self):
        # Clear the output once at the start
        clear_output(wait=True)
        super().create_bars()
    
    def update_display(self):
        # Capture the current state of all progress bars
        output = ""
        for i in range(self.nccd + 1):  # +1 for main bar
            output += f"\033[{i};0H"  # Move cursor to start of line
            if i == 0:
                output += str(self.main_bar)
            else:
                bar = self.progress_bars.get(i-1)
                if bar:
                    output += str(bar)
        
        # Clear and update the display
        clear_output(wait=True)
        print(output, end='')

