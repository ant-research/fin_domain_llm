import gradio as gr

from weaverbird.utils import get_kbs_list


class Demo:
    theme = gr.themes.Soft()

    block_css = """.importantButton {
            background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
            border: none !important;
        }
        .importantButton:hover {
            background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
            border: none !important;
        }
        a {
            cursor: pointer;
            text-decoration: none !important;
            align-items: center;
            justify-content: center;
            min-width: max-content;
            height: 24px;
            border-radius: 4px;
            box-sizing: border-box !important;
            padding: 0px 8px;
            color: #174ae4 !important;
            background-color: #d1dbfa;
            margin-inline-end: 6px;
        }

        a:hover {
            text-decoration: underline;
        }
        .message {
            width: auto !important;
        }
        .custom_btn {
            font-size: small !important;
        }
        .custom_btn_2 {
            font-size: 1.5em;
        }
        img {
            max-width: 100%;
            max-height:100%;
        }
        .container {
            display: grid;
            align-items: center;
            grid-template-columns: 1fr 1fr 1fr;
            column-gap: 1px;
        }
        .column {
            float: left;
            width: 33%;
        }

        #tracker {
            background-color: transparent;
            margin-inline-end: unset;
            padding: unset;
        }
        #tracker img {
            margin-left: 4em;
        }
        .custom_height {
            height: 2.5em;
        }
        .custom_width {
            max-width: 2.5em;
            min-width: 2.5em !important;
        }
        """

    en_title = """
    <h1 align="center" style="font-size: 3rem; color: #273746"> WeaverBird </h1>
    """

    en_sub_title = """
    <h4 align="center" style="font-size: 1rem; color: #CD5C5C">An Open and Light GPT for Finance</h4>
    """

    en_examples = [
        ["How will ChatGPT contribute to Nvidia's AI business in the short term?"],
        ["What does Tesla’s Elon Musk think of BYD rivalry"],
        ['What are the growth prospects for Microsoft Corporation in coming years']
    ]

    en_input_text = gr.Textbox(
        show_label=False,
        placeholder="""Ask a question and press ENTER. Be specific: use company names and specify times for best results.
    """,
        container=False)

    cn_title = """
    <h1 align="center" style="font-size: 3rem; color: #273746">织工鸟</h1>
    """

    cn_sub_title = """
    <h4 align="center" style="font-size: 1rem; color: #CD5C5C">一个开源且轻量级的金融领域GPT</h4>
    """

    cn_examples = [
        ['阿里巴巴的2023年Q1净利润多少?'],
        ['请写一篇公司简评，标题为比亚迪(002594.SZ)：2023年一季度业绩高速增长'],
        ["半夏资本李蓓的最新投资观点是什么"],
    ]

    cn_input_text = gr.Textbox(
        show_label=False,
        placeholder="输入问题，按回车键提交。请具体一些并包含公司名和时间段，这样效果会更好。",
        container=False)

    kb_root_dir = ''

    def __init__(self, **kwargs):
        self.theme = kwargs.pop('theme', self.theme)
        self.block_css = kwargs.pop('block_css', self.block_css)
        self.en_title = kwargs.pop('en_title', self.en_title)
        self.en_examples = kwargs.pop('en_examples', self.en_examples)
        self.cn_title = kwargs.pop('cn_title', self.cn_title)
        self.cn_examples = kwargs.pop('cn_examples', self.cn_examples)
        self.kb_root_dir = kwargs.pop('kb_root_dir', self.kb_root_dir)

    @staticmethod
    def set_example(example: list) -> dict:
        return gr.Textbox.update(value=example[0])

    @staticmethod
    def reset_history():
        return [], []

    def init_model(self):
        return

    def get_answer(self, query, chatbot, history, vs_name, search_engine):
        return

    def run(self):
        with gr.Blocks(css=self.block_css, theme=self.theme) as demo:
            chat_history = gr.State([])
            with gr.Tab('English'):
                gr.HTML(self.en_title)
                gr.HTML(self.en_sub_title)
                with gr.Tab('Chat'):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_engine = gr.Radio(["Off", "On"], label="Search Engine", value="On")
                            kb_list = get_kbs_list(self.kb_root_dir)
                            select_kb = gr.Dropdown(
                                kb_list,
                                label="Knowledge Base",
                                interactive=True,
                                value=kb_list[0] if len(kb_list) > 0 else None
                            )
                            with gr.Accordion("Try Asking About"):
                                example_text = gr.Examples(examples=self.en_examples,
                                                           fn=self.set_example,
                                                           inputs=self.en_input_text,
                                                           outputs=self.en_input_text,
                                                           label="Examples")
                        with gr.Column(scale=5):
                            with gr.Row():
                                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False, height=500)
                            with gr.Row():
                                empty_btn = gr.Button("🗑️ ",
                                                      elem_classes=['custom_height', 'custom_width', 'custom_btn_2'])
                                self.en_input_text.render()
                                sub_btn = gr.Button("➡️",
                                                    elem_classes=['custom_height', 'custom_width', 'custom_btn_2'])
                        sub_btn.click(self.get_answer,
                                      [self.en_input_text, chatbot, chat_history, select_kb, search_engine],
                                      [chatbot, chat_history, self.en_input_text])

                        empty_btn.click(self.reset_history, outputs=[chatbot, chat_history], show_progress=True)

                        self.en_input_text.submit(self.get_answer,
                                                  [self.en_input_text, chatbot, chat_history, select_kb, search_engine],
                                                  [chatbot, chat_history, self.en_input_text])

            with gr.Tab('中文'):
                gr.HTML(self.cn_title)
                gr.HTML(self.cn_sub_title)
                with gr.Tab('对话'):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_engine = gr.Radio(["关", "开"], label="搜索引擎", value="开")
                            kb_list = get_kbs_list(self.kb_root_dir)
                            select_kb = gr.Dropdown(
                                kb_list,
                                label="知识库",
                                interactive=True,
                                value=kb_list[0] if len(kb_list) > 0 else None
                            )
                            with gr.Accordion("可以尝试问这些问题"):
                                example_text = gr.Examples(examples=self.cn_examples,
                                                           fn=self.set_example,
                                                           inputs=self.cn_input_text,
                                                           outputs=self.cn_input_text,
                                                           label="参考问题")
                        with gr.Column(scale=5):
                            with gr.Row():
                                chatbot = gr.Chatbot(elem_id="chat-box", show_label=False, height=500)
                            with gr.Row():
                                empty_btn = gr.Button("🗑️ ",
                                                      elem_classes=['custom_height', 'custom_width', 'custom_btn_2'])
                                self.cn_input_text.render()
                                sub_btn = gr.Button("➡️",
                                                    elem_classes=['custom_height', 'custom_width', 'custom_btn_2'])
                        sub_btn.click(self.get_answer,
                                      [self.cn_input_text, chatbot, chat_history, select_kb, search_engine],
                                      [chatbot, chat_history, self.en_input_text])

                        empty_btn.click(self.reset_history, outputs=[chatbot, chat_history], show_progress=True)

                        self.cn_input_text.submit(self.get_answer,
                                                  [self.cn_input_text, chatbot, chat_history, select_kb, search_engine],
                                                  [chatbot, chat_history, self.cn_input_text])

        demo.queue(concurrency_count=50).launch(
            server_name='0.0.0.0',
            show_api=False,
            share=False,
            inbrowser=False)


def main():
    my_demo = Demo()

    my_demo.run()


if __name__ == '__main__':
    main()
