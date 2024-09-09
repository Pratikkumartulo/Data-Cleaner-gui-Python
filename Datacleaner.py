import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder , LabelEncoder, StandardScaler, MinMaxScaler
import io

dataframe = ""
file_path = ""
selected_option_var=""
datatype_var=""

def on_closing():
    root.destroy()

def browse_file():
    global file_path
    global dataframe
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        entry_field.config(state=tk.NORMAL)
        entry_field.delete(0, tk.END)
        entry_field.insert(0, file_path)
        entry_field.config(state=tk.DISABLED)
        dataframe = pd.read_csv(file_path)

def display_file():
    global file_path
    global dataframe
    if file_path:
        try:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, dataframe.to_string(index=False))
            text_widget.config(state=tk.DISABLED)
        except Exception as e:
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, f"Error: {e}")

def display_Null():
    try:
        global dataframe
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, dataframe.isnull().sum())
        text_widget.config(state=tk.DISABLED)

        for widget in dynamic_frame.winfo_children():
            widget.destroy()

        show_individual_button = tk.Button(dynamic_frame, text="Show Null Individual Percent", command=Show_Null_Individual)
        show_individual_button.pack(side=tk.TOP, padx=10, pady=5)

        show_whole_dataset_percent_button = tk.Button(dynamic_frame, text="Show Whole Dataset Null Percent", command=Show_Null_Percentage)
        show_whole_dataset_percent_button.pack(side=tk.TOP, padx=10, pady=5)

        show_graph_button = tk.Button(dynamic_frame, text="Show Null Values Graph", command=show_null_graph)
        show_graph_button.pack(side=tk.TOP, padx=10, pady=5)

        manage_null_button = tk.Button(dynamic_frame, text="Manage Null", command=manage_null)
        manage_null_button.pack(side=tk.TOP, padx=10, pady=5)
    except Exception as e:
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")

def Show_Null_Individual():
    try:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, (dataframe.isnull().sum() / dataframe.shape[0]) * 100)
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")

def Show_Null_Percentage():
    try:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Whole data has {(int(dataframe.isnull().sum().sum()) / (dataframe.shape[0] * dataframe.shape[1])) * 100}% Null values")
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")

def show_null_graph():
    global dataframe
    try:
        null_counts = dataframe.isnull()
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(null_counts, ax=ax)
        ax.set_title("Null Values in Each Column")

        # Clear previous graph from map_frame, if any
        for widget in map_frame.winfo_children():
            widget.destroy()

        # Display the graph inside map_frame
        canvas = FigureCanvasTkAgg(fig, master=map_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=8, pady=10)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

checkboxes = {}

def manage_null():
    global dataframe
    for widget in dynamic_frame.winfo_children():
        widget.destroy()

    options = [i for i in dataframe.columns.to_list()]
    rw = 0
    for option in options:
        var = tk.BooleanVar()
        checkbox = tk.Checkbutton(dynamic_frame, text=option, variable=var)
        checkbox.pack()
        checkboxes[option] = var

    submit_button = tk.Button(dynamic_frame, text="Delete columns", command=submit)
    submit_button.pack(side=tk.TOP, padx=10, pady=5)

    delete_null_btn = tk.Button(dynamic_frame, text="Delete all null rows", command=delete_all_null)
    delete_null_btn.pack(side=tk.TOP, padx=10, pady=5)

    bfill_null_btn = tk.Button(dynamic_frame, text="Backward fill rows", command=backward_fill_rows)
    bfill_null_btn.pack(side=tk.TOP, padx=10, pady=5)

    ffill_null_btn = tk.Button(dynamic_frame, text="Forward fill rows", command=forward_fill_rows)
    ffill_null_btn.pack(side=tk.TOP, padx=10, pady=5)

    fill_mode_null_btn = tk.Button(dynamic_frame, text="Fill mode", command=fill_mode_rows)
    fill_mode_null_btn.pack(side=tk.TOP, padx=10, pady=5)

    fill_mean_null_btn = tk.Button(dynamic_frame, text="Fill mean", command=fill_mean_rows)
    fill_mean_null_btn.pack(side=tk.TOP, padx=10, pady=5)

def submit():
    global dataframe
    selected_options = [option for option, var in checkboxes.items() if var.get()]
    dataframe.drop(columns=selected_options, inplace=True)
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, f"{', '.join(selected_options)} columns deleted")
    text_widget.config(state=tk.DISABLED)

def delete_all_null():
    global dataframe
    dataframe.dropna(inplace=True)
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, f"Null value rows deleted successfully!")
    text_widget.config(state=tk.DISABLED)

def backward_fill_rows():
    global dataframe
    dataframe.bfill(inplace=True)
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, f"Backward filled rows successfully!")
    text_widget.config(state=tk.DISABLED)

def forward_fill_rows():
    global dataframe
    dataframe.ffill(inplace=True)
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, f"Forward filled rows successfully!")
    text_widget.config(state=tk.DISABLED)

def fill_mode_rows():
    global dataframe
    for i in dataframe.select_dtypes(include="object").columns:
        dataframe[i].fillna(dataframe[i].mode()[0], inplace=True)
    text_widget.config(state=tk.NORMAL)
    text_widget.delete(1.0, tk.END)
    text_widget.insert(tk.END, f"Filled rows with mode successfully!")
    text_widget.config(state=tk.DISABLED)

def fill_mean_rows():
    global dataframe
    try:
        numeric_columns = dataframe.select_dtypes(include=["int64", "float64"]).columns
        si = SimpleImputer(strategy="mean")
        dataframe[numeric_columns] = si.fit_transform(dataframe[numeric_columns])
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Filled rows with mean successfully!")
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def encode():
    for widget in map_frame.winfo_children():
            widget.destroy()
    global dataframe
    try:
        for widget in dynamic_frame.winfo_children():
            widget.destroy()
        one_hot_encode = tk.Button(dynamic_frame,text="One hot encode",command=one_hot_encode_options)
        one_hot_encode.pack(side=tk.TOP)
        label_encode = tk.Button(dynamic_frame,text="Label encode",command=label_encode_options)
        label_encode.pack(side=tk.TOP)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def label_encode_options():
     global dataframe, checkboxes
     checkboxes.clear()
     for widget in map_frame.winfo_children():
            widget.destroy()
     options = [i for i in dataframe.select_dtypes(include="object").columns.to_list()]
     for option in options:
         var = tk.BooleanVar()
         checkbox = tk.Checkbutton(map_frame, text=option, variable=var)
         checkbox.pack(side=tk.TOP, padx=10, pady=5)
         checkboxes[option] = var
     label_btn = tk.Button(map_frame,text="Encode it",command=label_ecodeit)
     label_btn.pack(side=tk.TOP)

def one_hot_encode_options():
    global dataframe, checkboxes
    checkboxes.clear()
    for widget in map_frame.winfo_children():
            widget.destroy()
    options = [i for i in dataframe.select_dtypes(include="object").columns.to_list()]
    for option in options:
        if len(dataframe[option].unique()) <= 2:
                var = tk.BooleanVar()
                checkbox = tk.Checkbutton(map_frame, text=option, variable=var)
                checkbox.pack(side=tk.TOP, padx=10, pady=5)
                checkboxes[option] = var
    one_hot_btn = tk.Button(map_frame,text="Encode it",command=One_hot_ecodeit)
    one_hot_btn.pack(side=tk.TOP)
    
def label_ecodeit():
    global dataframe, checkboxes
    for widget in dynamic_frame.winfo_children():
        widget.destroy()
    for widget in map_frame.winfo_children():
            widget.destroy()
    selected_options = [option for option, var in checkboxes.items() if var.get()]
    
    if selected_options:
        le = LabelEncoder()
        for column in selected_options:
            dataframe[column] = le.fit_transform(dataframe[column])
        checkboxes.clear()
        
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, "Label Encoding applied successfully!")
        text_widget.config(state=tk.DISABLED)


def One_hot_ecodeit():
    global dataframe, checkboxes
    for widget in dynamic_frame.winfo_children():
            widget.destroy()
    for widget in map_frame.winfo_children():
            widget.destroy()
    selected_options = [option for option, var in checkboxes.items() if var.get()]
    if selected_options: 
        en_code = dataframe[selected_options]
        ohe = OneHotEncoder(drop="first")
        ar = ohe.fit_transform(en_code).toarray()
        ohe_df = pd.DataFrame(ar, columns=ohe.get_feature_names_out(selected_options))
        dataframe = pd.concat([dataframe, ohe_df], axis=1)
        dataframe.drop(selected_options, axis=1, inplace=True)
        checkboxes.clear() 
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, "Encoded Successfully !!")
        text_widget.config(state=tk.DISABLED)
        for widget in dynamic_frame.winfo_children():
            widget.destroy()
    else:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, "No columns selected for encoding!")
        text_widget.config(state=tk.DISABLED)

def data_info():
    global file_path
    global dataframe
    for widget in map_frame.winfo_children():
            widget.destroy()
    if file_path:
        try:
            buffer = io.StringIO()
            dataframe.info(buf=buffer)
            info_str = buffer.getvalue()

            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, info_str)
            text_widget.config(state=tk.DISABLED)
        except Exception as e:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, f"Error: {e}")
            text_widget.config(state=tk.DISABLED)

def describe():
    global dataframe
    for widget in map_frame.winfo_children():
            widget.destroy()
    try:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, dataframe.describe())
        text_widget.config(state=tk.DISABLED)

    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def outliers():
    global dataframe, checkboxes,selected_option_var
    try:
        for widget in dynamic_frame.winfo_children():
            widget.destroy()
        for widget in map_frame.winfo_children():
            widget.destroy()
        selected_option_var = tk.StringVar()
        options = [i for i in dataframe.describe().columns.to_list()]
        for option in options:
            radio_button = tk.Radiobutton(dynamic_frame, text=option, variable=selected_option_var, value=option)
            radio_button.pack(side=tk.TOP, padx=10, pady=5)
        selected_option_var.set(options[0])
        Box_plot = tk.Button(dynamic_frame,text="Box plot",command=box_plot)
        Box_plot.pack(side=tk.TOP)
        Dis_plot = tk.Button(dynamic_frame,text="Distplot",command=dis_plot)
        Dis_plot.pack(side=tk.TOP)
        IQR_btn = tk.Button(dynamic_frame,text="IQR method",command=iqr_outlier)
        IQR_btn.pack(side=tk.TOP)
        Z_Sc_btn = tk.Button(dynamic_frame,text="Z-Sc method",command=Z_Sc_outlier)
        Z_Sc_btn.pack(side=tk.TOP)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def box_plot():
    global dataframe, selected_option_var
    try:
        selected_column = selected_option_var.get()
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.boxplot(data=dataframe, x=selected_column, ax=ax)
        ax.set_title(f"Box Plot of {selected_column}")
        for widget in map_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=map_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=8, pady=10)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def dis_plot():
    global dataframe, selected_option_var
    try:
        selected_column = selected_option_var.get()
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.distplot(dataframe[selected_column])
        ax.set_title(f"DisPlot of {selected_column}")
        for widget in map_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=map_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(padx=8, pady=10)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def iqr_outlier():
    global dataframe, selected_option_var
    try:
        selected_column = selected_option_var.get()
        q1 = dataframe[selected_column].quantile(0.25)
        q3 = dataframe[selected_column].quantile(0.75)
        IQR = q3-q1
        min_range = q1-(1.5*IQR)
        max_range = q3+(1.5*IQR)
        dataframe = dataframe[dataframe[selected_column]<=max_range]
        dataframe = dataframe[dataframe[selected_column]>=min_range]
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"{selected_column} column's outlier removed")
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def  Z_Sc_outlier():
    global dataframe, selected_option_var
    try:
        selected_column = selected_option_var.get()
        min_range = dataframe[selected_column].mean() - (3*dataframe[selected_column].std())
        max_range = dataframe[selected_column].mean() + (3*dataframe[selected_column].std())
        dataframe = dataframe[dataframe[selected_column]<=max_range]
        dataframe = dataframe[dataframe[selected_column]>=min_range]
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"{selected_column} column's outlier removed")
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def Scalling():
    global dataframe, checkboxes,selected_option_var
    try:
        for widget in dynamic_frame.winfo_children():
            widget.destroy()
        for widget in map_frame.winfo_children():
            widget.destroy()
        selected_option_var = tk.StringVar()
        options = [i for i in dataframe.describe().columns.to_list()]
        for option in options:
            radio_button = tk.Radiobutton(dynamic_frame, text=option, variable=selected_option_var, value=option)
            radio_button.pack(side=tk.TOP, padx=10, pady=5)
        selected_option_var.set(options[0])
        Sscaller_btn = tk.Button(dynamic_frame,text="Standard scaller",command=MinMax_scaller)
        Sscaller_btn.pack(side=tk.TOP)
        Mmscaller_btn = tk.Button(dynamic_frame,text="Normalize scaller",command=MinMax_scaller)
        Mmscaller_btn.pack(side=tk.TOP)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def feature_standard():
    global dataframe, selected_option_var
    try:
        selected_column = selected_option_var.get()
        sc = StandardScaler()
        sc.fit(dataframe[[selected_column]])
        ar = sc.transform(dataframe[[selected_column]])
        Standard = pd.DataFrame(ar,columns=[selected_column])
        dataframe.drop(selected_column, axis=1, inplace=True)
        dataframe = pd.concat([dataframe, Standard], axis=1)
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"{selected_column} scalled succesfully")
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def MinMax_scaller():
    global dataframe, selected_option_var
    try:
        selected_column = selected_option_var.get()
        mm =MinMaxScaler()
        mm.fit(dataframe[[selected_column]])
        ar = mm.transform(dataframe[[selected_column]])
        Standard = pd.DataFrame(ar,columns=[selected_column])
        dataframe.drop(selected_column, axis=1, inplace=True)
        dataframe = pd.concat([dataframe, Standard], axis=1)
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"{selected_column} scalled succesfully")
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def duplicates():
    global file_path
    global dataframe
    if file_path:
        try:
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, dataframe.duplicated())
            text_widget.config(state=tk.DISABLED)
            for widget in dynamic_frame.winfo_children():
                widget.destroy()
            drop_dup_btn = tk.Button(dynamic_frame,text="Drop duplicate",command=drop_dup)
            drop_dup_btn.pack(side=tk.TOP)
        except Exception as e:
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, f"Error: {e}")
def drop_dup():
    global dataframe, selected_option_var
    try:
        dataframe.drop_duplicates(inplace=True)
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Duplicate drop succesfully")
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def more_option():
    global dataframe, selected_option_var
    try:
         for widget in dynamic_frame.winfo_children():
                widget.destroy()
         for widget in map_frame.winfo_children():
                widget.destroy()
         selected_option_var = tk.StringVar()
         options = [i for i in dataframe.columns.to_list()]
         for option in options:
            radio_button = tk.Radiobutton(dynamic_frame, text=option, variable=selected_option_var, value=option)
            radio_button.pack(side=tk.TOP, padx=10, pady=5)
         selected_option_var.set(options[0])
         replace_btn = tk.Button(dynamic_frame,text="Replace",command=replace_option)
         replace_btn.pack(side=tk.TOP)
         changedt_btn = tk.Button(dynamic_frame,text="Change datatype",command=datatype_option)
         changedt_btn.pack(side=tk.TOP)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def replace_option():
    global dataframe, selected_option_var, input_widgets
    try: 
        for widget in map_frame.winfo_children():
            widget.destroy()
        
        selected_column = selected_option_var.get()
        unique_values = dataframe[selected_column].unique().tolist()
        input_widgets = {}
        
        for value in unique_values:
            label = tk.Label(map_frame, text=f"Replace '{value}' with:")
            label.pack()
            entry = tk.Entry(map_frame)
            entry.pack()
            input_widgets[value] = entry
        replace_btn = tk.Button(map_frame, text="Replace", command=replace_values)
        replace_btn.pack(side=tk.TOP)
    
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def replace_values():
    global dataframe, selected_option_var, input_widgets
    try:
        selected_column = selected_option_var.get()
        for original_value, entry_widget in input_widgets.items():
            new_value = entry_widget.get()
            if new_value:
                dataframe[selected_column].replace(original_value, new_value, inplace=True)
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, "Values replaced successfully!")
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def datatype_option():
    global dataframe, selected_option_var, datatype_var
    try: 
        for widget in map_frame.winfo_children():
            widget.destroy()
        
        selected_column = selected_option_var.get()
        datatype = dataframe[selected_column].dtype
        label = tk.Label(map_frame, text=f"Current Data Type: {datatype}")
        label.pack()
        datatype_var = tk.StringVar(map_frame)
        datatype_var.set("Select new data type")
        dropdown_menu = tk.OptionMenu(map_frame, datatype_var, "object", "int", "float")
        dropdown_menu.pack()
        apply_btn = tk.Button(map_frame, text="Change Data Type", command=apply_datatype_change)
        apply_btn.pack(side=tk.TOP)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def apply_datatype_change():
    global dataframe, selected_option_var, datatype_var
    try:
        selected_column = selected_option_var.get()
        new_datatype = datatype_var.get()
        if new_datatype == "object":
            dataframe[selected_column] = dataframe[selected_column].astype(str)
        elif new_datatype == "int":
            dataframe[selected_column] = dataframe[selected_column].astype(int)
        elif new_datatype == "float":
            dataframe[selected_column] = dataframe[selected_column].astype(float)
        else:
            raise ValueError("Invalid data type selected.")
        
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Data type of '{selected_column}' changed to {new_datatype}.")
        text_widget.config(state=tk.DISABLED)
    except ValueError as ve:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {ve}")
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)
def export_option():
    global dataframe
    try:
        for widget in dynamic_frame.winfo_children():
            widget.destroy()
        file_label = tk.Label(dynamic_frame, text="Enter file name:")
        file_label.pack()
        
        file_name_var = tk.StringVar()
        file_name_entry = tk.Entry(dynamic_frame, textvariable=file_name_var)
        file_name_entry.pack()
        file_type_var = tk.StringVar(dynamic_frame)
        file_type_var.set("Select file type")

        file_type_dropdown = tk.OptionMenu(dynamic_frame, file_type_var, "CSV", "Excel")
        file_type_dropdown.pack()
        folder_label = tk.Label(dynamic_frame, text="Select folder:")
        folder_label.pack()
        
        folder_path_var = tk.StringVar()
        folder_path_entry = tk.Entry(dynamic_frame, textvariable=folder_path_var)
        folder_path_entry.pack()

        browse_button = tk.Button(dynamic_frame, text="Browse", command=lambda: browse_folder(folder_path_var))
        browse_button.pack()
        export_button = tk.Button(dynamic_frame, text="Export", command=lambda: export_file(file_name_var.get(), file_type_var.get(), folder_path_var.get()))
        export_button.pack()
    
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)

def browse_folder(folder_path_var):
    """Open a file dialog to select a folder and update the folder path entry."""
    folder_selected = filedialog.askdirectory()
    folder_path_var.set(folder_selected)

def export_file(file_name, file_type, folder_path):
    global dataframe
    try:
        if not file_name:
            raise ValueError("File name cannot be empty.")
        if file_type not in ["CSV", "Excel"]:
            raise ValueError("Please select a valid file type (CSV or Excel).")
        if not folder_path or not os.path.isdir(folder_path):
            raise ValueError("Please select a valid folder.")
        if file_type == "CSV":
            full_file_path = os.path.join(folder_path, f"{file_name}.csv")
            dataframe.to_csv(full_file_path, index=False)
        elif file_type == "Excel":
            full_file_path = os.path.join(folder_path, f"{file_name}.xlsx")
            dataframe.to_excel(full_file_path, index=False)
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"File successfully exported to: {full_file_path}")
        text_widget.config(state=tk.DISABLED)
    except ValueError as ve:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {ve}")
        text_widget.config(state=tk.DISABLED)
    except Exception as e:
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, f"Error: {e}")
        text_widget.config(state=tk.DISABLED)



root = tk.Tk()
root.title("Data Analysis Tool")
root.geometry("800x600")


start_frame = tk.Frame(root)
start_frame.grid(row=0,column=0)

entry_field = tk.Entry(start_frame, width=80, state=tk.DISABLED)
entry_field.pack(side=tk.LEFT, padx=10, pady=10)

browse_button = tk.Button(start_frame, text="Select File", command=browse_file)
browse_button.pack(side=tk.LEFT, padx=10, pady=10)

display_button = tk.Button(start_frame, text="Display Data", command=display_file)
display_button.pack(side=tk.LEFT, padx=10, pady=10)

button_frame = tk.Frame(root,width=100)
button_frame.grid(row=1,column=1,padx=10, pady=10)

data_info_button = tk.Button(button_frame,text="Data info",command=data_info)
data_info_button.pack(side=tk.TOP, padx=10, pady=5)

show_null_button = tk.Button(button_frame, text="Show Null", command=display_Null)
show_null_button.pack(side=tk.TOP,padx=10, pady=5)

encode_button = tk.Button(button_frame, text="Encode", command=encode)
encode_button.pack(side=tk.TOP,padx=10, pady=5)

describe_button = tk.Button(button_frame, text="Describe", command=describe)
describe_button.pack(side=tk.TOP,padx=10, pady=5)

outlier_button = tk.Button(button_frame, text="Outliers", command=outliers)
outlier_button.pack(side=tk.TOP,padx=10, pady=5)

Scalling_button = tk.Button(button_frame, text="Scalling", command=Scalling)
Scalling_button.pack(side=tk.TOP,padx=10, pady=5)

duplicates_button = tk.Button(button_frame, text="Show duplicates", command=duplicates)
duplicates_button.pack(side=tk.TOP,padx=10, pady=5)

more_op_button = tk.Button(button_frame, text="More options", command=more_option)
more_op_button.pack(side=tk.TOP,padx=10, pady=5)

export_button = tk.Button(button_frame, text="Export options", command=export_option)
export_button.pack(side=tk.TOP,padx=10, pady=5)

dynamic_frame = tk.Frame(root,height=20,width=20)
dynamic_frame.grid(row=1, column=2,sticky="nw")

map_frame = tk.Frame(root)
map_frame.grid(row=1,column=4)

h_scrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL)
h_scrollbar.grid(row=2, column=0, sticky='ew', padx=10, pady=5)  

text_widget = tk.Text(root, height=40, wrap="none", xscrollcommand=h_scrollbar.set)
text_widget.grid(row=1, column=0, padx=10, pady=10)

h_scrollbar.config(command=text_widget.xview)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
