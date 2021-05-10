import io
import uuid
import base64
from matplotlib import pyplot as plt

## Floating box
"""
Author: Frank Zalkow
License: The MIT license, https://opensource.org/licenses/MIT
"""


class FloatingBox(object):
    def __init__(self, align='middle'):
        # https://www.w3schools.com/cssref/pr_pos_vertical-align.asp

        self.class_name = f'floating-box-fmp-{uuid.uuid4()}'
        self.html = f"""
        <style>
        .{self.class_name} {{
        display: inline-block;
        margin: 10px;
        vertical-align: {align};
        }}
        </style>
        """

    def add_fig(self, fig):

        Bio = io.BytesIO()
        fig.canvas.print_png(Bio)

        # encode the bytes as string using base 64
        img = base64.b64encode(Bio.getvalue()).decode()
        self.html += (
            f'<div class="{self.class_name}">' +
            f'<img src="data:image/png;base64,{img}\n">' +
            '</div>')

        plt.close(fig)

    def add_html(self, html):

        self.html += (
            f'<div class="{self.class_name}">' +
            f'{html}' +
            '</div>')

    def show(self):
        display(HTML(self.html))
