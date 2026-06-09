import numpy as np
import plotly.graph_objects as go
import subprocess

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
z= np.linspace(-5,5,100)

X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(
    data=[go.Surface(x=X, y=Y, z=Z)]
)

fig.show()
fig.write_html("scan.html")
subprocess.run("explorer.exe scan.html", shell=True)