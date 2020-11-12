import tkinter
import numpy as np
from tkinter import ttk
from tkinter import filedialog

from pathlib import Path
import sys

# get inference helper from runtime eample
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / 'examples'))
from runtime import InferenceHelper, vrt

try:
    import tkinter
    import ttk
except ImportError as e:
    raise SystemExit("cant import tkinter which is needed for this example;\
        you can install with `pip install pyttk tk`"
    )

import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)

class ModelExplorer:
    """Explore existing Classification & Detection model interactively
    """    
    def __init__(self, title):
        """initialize visualization

        Args:
            title (str): title of visualizer
        """        
        # filenames
        self.filenames = []
        self.runtime = [] # running time stats
        self.title = title
        self.model = None
        self.model_path = None

        self.init_ui()

    def init_ui(self):
        """initialize UI
        """        
        root = tkinter.Tk()
        root.title(self.title)
        # Add a grid
        mainframe = tkinter.Frame(root)
        sticky = (tkinter.N,tkinter.W,tkinter.E,tkinter.S)
        mainframe.grid(column=0, row=0, sticky=sticky)
        mainframe.columnconfigure(0, weight=1)
        mainframe.rowconfigure(0, weight=1)
        mainframe.pack(pady=10, padx=100)
        self.root = root
        self.mainframe = mainframe

        # setup menu for open directory
        menubar = tkinter.Menu(root)
        filemenu = tkinter.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open Directory", command=self.open_directory)
        filemenu.add_separator()
        filemenu.add_command(label="Load Model", command=self.load_model)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.filemenu = filemenu
        self.menubar = menubar
        root.configure(menu=self.menubar)

        # runtime device selection
        runtime_var = tkinter.StringVar(root)
        # dict of dict, get values only
        choices = list(vrt.model_runtime_map.values())
        # get keys from list of dict
        choices = list(map(lambda x: list(x.keys()), choices))
        # flatten
        choices = [str(l) for sublist in choices for l in sublist]
        choices = list(set(choices))
        runtime_var.set(choices[0]) # set the default option
        runtime_label = ttk.Label(mainframe, text="Runtime").grid(row=0, column=0)
        self.runtime_var = runtime_var
        # note that width is not automatically adjusted
        runtime_combobox = ttk.Combobox(mainframe, textvariable=runtime_var, values=choices, width=90)
        runtime_combobox.grid(row=0, column=1)
        self.runtime_combobox = runtime_combobox

        reload_button = ttk.Button(mainframe, text ="Reload", command=self.reload).grid(row=0, column=2)
        self.reload_button = reload_button

        # dropdown
        dropdown_var = tkinter.StringVar(root)
        # Dictionary with options
        choices = list([str(i) for i in range(len(self.filenames))])
        dropdown_var.set('0') # set the default option
        label = ttk.Label(mainframe, text="Filenames").grid(row=1, column=1)
        dropdown_var.trace('w', self.change_dropdown)
        self.dropdown_var = dropdown_var

        # note that width is not automatically adjusted
        combobox = ttk.Combobox(mainframe, textvariable=dropdown_var, values=choices, width=90)
        combobox.grid(row=2, column=1)
        self.combobox = combobox

        next_button = ttk.Button(mainframe, text ="Next", command=self.next).grid(row=2, column=2)
        prev_button = ttk.Button(mainframe, text ="Prev", command=self.prev).grid(row=2, column=0)
        self.next_button = next_button
        self.prev_button = prev_button

        self.label = label

        # setup figure
        figure = plt.figure()
        figure.tight_layout()
        self.figure = figure

        # plotter
        canvas = FigureCanvasTkAgg(self.figure, master=root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        self.canvas = canvas
        self.toolbar = toolbar

        # TODO: add ui for additional args
        self.additional_args = dict(
            score_threshold=0.9,
            iou_threshold=0.1,
        )
    
    def quit(self):
        """quit gui program
        """        
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent
    
    def load_model(self):
        """prompt filename dialog and then load model
        """
        filename = filedialog.askopenfilename()
        if filename:
            self.model_path = filename
            self.reload()
    
    def reload(self):
        """load model from saved model_path & runtime
        """
        if Path(self.model_path).exists() :
            kwargs = dict(
                model_path=self.model_path,
                runtime=self.runtime_var.get(),
            )
            print('loading model, this may take a while')
            self.model = InferenceHelper.create_runtime_model(**kwargs)
            print('loading model done:', type(self.model.model))
            self.runtime = []
            title = f"{self.title}: {self.model_path}"
            self.root.title(title)
        else:
            print(f'{self.model_path} doesnt exists')

    def open_directory(self):
        """prompt directory dialog and then update existing filenames
        """
        dirname = filedialog.askdirectory()
        self.filenames = list(Path(dirname).iterdir())
        self.combobox['values'] = self.filenames
        self.combobox.update()
        self.dropdown_var.set(self.filenames[0])
    
    def inference(self, kwargs: dict):
        """run inference on loaded model

        Args:
            kwargs (dict): arguments to model call
        """        
        if self.model is None:
            print('model is empty, load model first!')
        else:
            kwargs.update(visualize=True) # force to visualize
            print(kwargs)
            results = self.model(**kwargs)
            self.runtime.append(results['runtime'])
            vis = results['visualization']
            if len(vis) > 1:
                # image must be on the same shape before stacking
                shape = vis[0].shape[-2::-1]
                vis = list(map(lambda x: cv2.resize(x, shape), vis))
            # simply stack visualization accross batch
            image = np.vstack(vis)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gs = self.figure.add_gridspec(1,1)
            img_ax = self.figure.add_subplot(gs[0:,0:2])
            img_ax.imshow(image)
            print("prediction results\n", results['prediction'])
            print("runtime={}s; median={}; mean={}; min={}; max={};".format(
                results['runtime'], np.median(self.runtime),
                np.mean(self.runtime), np.min(self.runtime), np.max(self.runtime),
            ))

    def change_dropdown(self,*args):
        """callback to be called when user change dropdown menu
        """        
        key = Path(self.dropdown_var.get())
        self.figure.clear()
        if key in self.filenames:
            kwargs = dict(
                images=str(key),
                **self.additional_args
            )
            self.inference(kwargs)
            self.figure.tight_layout()
            self.canvas.draw()
    
    def next(self):
        """callback to be called when user click next button
        """        
        key = self.filenames.index(Path(self.dropdown_var.get())) + 1
        if key >= len(self.filenames):
            key = 0
        key = str(self.filenames[key])
        self.dropdown_var.set(key)
    
    def prev(self):
        """callback to be called when uesr click prev button
        """        
        key = self.filenames.index(Path(self.dropdown_var.get())) - 1
        if key < 0:
            key = 0
        key = str(self.filenames[key])
        self.dropdown_var.set(key)

    def mainloop(self):
        """blocking call to actual gui render
        """
        self.root.mainloop()

def main(args):
    gui = ModelExplorer('model explorer')
    gui.mainloop()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)