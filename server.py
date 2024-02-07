from flask import Flask, make_response, render_template
import sys
import pandas as pd
import openai
import traceback
import nltk
from openai import OpenAI
import httpx
from nltk.tokenize import sent_tokenize, word_tokenize
from flask import Flask, request
from docx import Document
import os

is_local_run = True
app = Flask(__name__)
key = os.getenv('openai_key')
if is_local_run:
    client = OpenAI(api_key=key,
                    http_client=httpx.Client(
                        proxies="http://127.0.0.1:10809",
                        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                    ),)
else:
     client = OpenAI(api_key=key)

test_result = '''**********
| Theme | Description | Quotes | Participant Count |
---|---|---|---|
| Election Strategy | Participants discussing different strategies and tactics for the upcoming election. | "那我投1號，就可以全部都投到了啊" | 1 |
| Humor | Participants sharing jokes and humorous references. | "忍不住想到那張湯姆貓與傑利鼠的梗圖…急了😂" | 1 |
| Opinion Polls | Participants discussing the results of opinion polls and their implications. | "但郭董最新的 3000 樣本民調結果中 柯賴 在誤差範圍內而 您 差距巨大" | 1 |
| Voter Support | Participants expressing their support for a specific candidate or party. | "我是老國民黨員，這次真的被年輕人的熱血所感動，這次我投柯" | 1 |
| Policy Goals | Participants expressing their desires for a better society and improvements in various areas. | "我們真心想贏得民生富裕 想有和平環境生活" | 1 |
| Repeating Tactics | Participants criticizing the repetition of past political tactics. | "2000年把宋楚瑜做掉的爛招，過了24年還想再玩一次" | 1 |
| Professionalism | Participants discussing the professionalism and expertise of a candidate. | "侯友宜先生總是是默默把事做好,卻不張揚" | 1 |
| Anti-incumbent Sentiment | Participants expressing their desire to remove the incumbent political party. | "支持郭董的理念「下架民進黨」" | 1 |
| Negative Campaigning | Participants criticizing negative campaigning and political tactics. | "反對把戲" | 1 |
| Land Development | Participants discussing land development issues. | "地產大亨在獵地" | 1 |
**********'''

data_type_dict = {'1':'Interview','2':'Focus Group','3':'Social Media Posts'}
prompts = {
                "Interview": "You need to analyze an dataset of interviews. \
                \nPlease identify the top {num_themes} key themes from the interview and organize the results in a structured table format. \
                \nThe table should includes these items:\
                \n- 'Theme': Represents the main idea or topic identified from the interview.\
                \n- 'Description': Provides a brief explanation or summary of the theme.\
                \n- 'Quotes': Contains direct quotations from participants that support the identified theme.\
                \n- 'Participant Count': Indicates the number of participants who mentioned or alluded to the theme.\
                \nThe table should be formatted as follows: \
                \nEach column should be separated by a '|' symbol, and there should be no extra '|' symbols within the data. Each row should end with '---'. \
                \nThe whole table should start with '**********' and end with '**********'.\
                \nColumns: | 'Theme' | 'Description' | 'Quotes' | 'Participant Count' |. \
                \nEnsure each row of the table represents a distinct theme and its associated details. Aggregate the counts for each theme to show the total number of mentions across all participants.",
            "Focus Group": "You need to analyze an dataset of a focus group. \
                \nPlease identify the {num_themes} most common key themes from the interview and organize the results in a structured table format. \
                \nThe table should includes these items:\
                \n- 'Theme': Represents the main idea or topic identified from the interview.\
                \n- 'Description': Provides a brief explanation or summary of the theme.\
                \n- 'Quotes': Contains direct quotations from participants that support the identified theme.\
                \n- 'Participant Count': Indicates the number of participants who mentioned or alluded to the theme. Please ensure this count reflects the actual number of participants who discussed each theme.\
                \nThe table should be formatted strictly as follows: \
                \nThe table should have 4 columns only.\
                \nEach column should be separated by a '|' symbol, and there should be no extra '|' symbols within the data. Each row should end with '---'. \
                \nStart the table with '**********'.\
                \nThe header row should be: | 'Theme' | 'Description' | 'Quotes' | 'Participant Count' | \
                \nFollowed by a row of '|---|---|---|---|'.\
                \nEnd the table with '**********'.\
                \nEach subsequent row should represent a theme and its details, with columns separated by '|'.\
                \nEnsure each row of the table represents a distinct theme and its associated details.",
            "Social Media Posts": "You need to analyze an dataset of Social Media Posts. \
                \nPlease identify the top {num_themes} key themes from the interview and organize the results in a structured table format. \
                \nThe table should includes these items:\
                \n- 'Theme': Represents the main idea or topic identified from the interview.\
                \n- 'Description': Provides a brief explanation or summary of the theme.\
                \n- 'Quotes': Contains direct quotations from participants that support the identified theme.\
                \n- 'Participant Count': Indicates the number of participants who mentioned or alluded to the theme.\
                \nThe table should be formatted as follows: \
                \nEach column should be separated by a '|' symbol, and there should be no extra '|' symbols within the data. Each row should end with '---'. \
                \nThe whole table should start with '**********' and end with '**********'.\
                \nColumns: | 'Theme' | 'Description' | 'Quotes' | 'Participant Count' |. \
                \nEnsure each row of the table represents a distinct theme and its associated details."
            }

def analyze_merged_responses(merged_responses,num_themes):
        # Construct a new prompt for the merged responses
        new_prompt = "This is the result of a thematic analysis of several parts of the dataset. Now, summarize the same themes to generate a new table. \
        \nPlease identify the {num_themes} most common key themes from the interview and organize the results in a structured table format. \
        \nThe table should include the following columns:\
        \n'Theme': Represents the main idea or topic identified from the interview.\
        \n'Description': Provides a brief explanation or summary of the theme.\
        \n'Quotes': Contains direct quotations from participants that support the identified theme.\
        \n'Participant Count': Indicates the number of participants who mentioned or alluded to the theme. Ensure this count reflects the actual number of participants who discussed each theme.\
        \nThe table should be formatted strictly as follows:\
\n- Start the table with '**********'.\
\n- The header row should be: | 'Theme' | 'Description' | 'Quotes' | 'Participant Count' |\
\n- Followed by a row of '|---|---|---|---|'.\
\n- Each subsequent row should represent a theme and its details, with columns separated by '|'.\
\n- Each row should end with '---'.\
\n- End the table with '**********'.\
\nEnsure each row of the table represents a distinct theme and its associated details. \
\nAnalyze the following merged responses: " + merged_responses
        # self.display_prompt(new_prompt)
        new_prompt=new_prompt.format(num_themes=num_themes)
        try:
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": new_prompt}
            ])
            response_content = response.choices[0].message.content
            
            return response_content
            # Display the final analysis
            # self.text_area.moveCursor(QTextCursor.End)
            # self.text_area.append("Final Analysis:\n" + response_content)
        except openai.error.OpenAIError as e:
            print(f"OpenAI Error: {str(e)}")
            # QMessageBox.critical(self, "Error", f"Failed to analyze merged responses. OpenAI Error: {str(e)}")
            return f"Failed to analyze merged responses. OpenAI Error: {str(e)}"
        except Exception as e:
            print(f"Other Error: {str(e)}")
            # QMessageBox.critical(self, "Error", f"Failed to analyze merged responses. Other Error: {str(e)}")
            return f"Failed to analyze merged responses. Other Error: {str(e)}"

def call_chatgpt(num_themes,prompt,data_content,saved_segments,custom_prompt=''):    
    # num_themes = self.key_themes_spinbox.value()
    # prompt = self.custom_prompt_entry.text().strip()
    # if not prompt:
    #     prompt = self.preset_prompts.currentText().format(num_themes=num_themes)
    
    
    prompt=prompt.format(num_themes=num_themes)
    prompt+=' '+custom_prompt

    # Combine the dataset and the prompt into a single message
    #combined_message = self.data_content + "\n\n" + prompt
    all_responses=[]
    if len(saved_segments) > 1:
        
        for segment in saved_segments:
        # Construct the full prompt for this segment
            combined_message = segment + "\n\n" + prompt  # Use the same prompt for each segment
            # Display the prompt being sent to the API
            # self.display_prompt(combined_message)
            # Send the segment to the API
            try:
                response_content=combined_message
                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": combined_message}
                ])
                response_content = response.choices[0].message.content
                all_responses.append(response_content) # save the response
                
                # Check if the response is close to the token limit
                if len(response_content.split()) > 4000:  # This is an arbitrary number, adjust as needed
                    # QMessageBox.warning(self, "Warning", "The response might be truncated due to token limits.")
                    print("Warning", "The response might be truncated due to token limits.")
            except openai.error.OpenAIError as e:
                print(f"OpenAI Error: {str(e)}")
            except Exception as e:
                print(f"Other Error: {str(e)}")
        # After processing all segments, merge the responses and analyze again
        merged_responses = "\n".join(all_responses)
        # analyze_merged_responses(merged_responses)
        return analyze_merged_responses(merged_responses,num_themes)
    else:
        combined_message = data_content + "\n\n" + prompt
        # Display the prompt being sent to the API
        # display_prompt(combined_message)
         # Send the segment to the API
        try:
            response_content=combined_message

            response =  client.chat.completions.create(model="gpt-3.5-turbo", messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": combined_message}
            ])
            response_content = response.choices[0].message.content
            all_responses.append(response_content) # save the response
            
            # Check if the response is close to the token limit
            if len(response_content.split()) > 4000:  # This is an arbitrary number, adjust as needed
                print("Warning", "The response might be truncated due to token limits.")
            
            # self.text_area.moveCursor(QTextCursor.End)
            # self.text_area.append("Response:\n" + response_content)
            return all_responses
        except openai.APIError  as e:
            print(f"OpenAI Error: {str(e)}")
            # QMessageBox.critical(self, "Error", f"Failed to call ChatGPT API. OpenAI Error: {str(e)}")
        except Exception as e:
            print(f"Other Error: {str(e)}")
            # QMessageBox.critical(self, "Error", f"Failed to call ChatGPT API. Other Error: {str(e)}")
                                          
def parse_response_to_csv(response):
    # Split the response into lines
    lines = response.strip().split("\n")
    # Find all occurrences of the table delimiter
    delimiter_indices = [i for i, line in enumerate(lines) if line.strip() == "**********"]
    # If there are fewer than two delimiters, return an empty list
    if len(delimiter_indices) < 2:
        return []
    # Use the first and last occurrences of the delimiter to identify the start and end of the table
    start_index, end_index = delimiter_indices[0], delimiter_indices[-1]
    # Extract the table content
    table_content = lines[start_index+1:end_index]
    # Split each line into columns based on the '|' character
    parsed_data = [line.split("|")[1:-1] for line in table_content if line.strip()]  # Exclude the first and last elements
    # Remove whitespace from each cell
    parsed_data = [[cell.strip() for cell in row] for row in parsed_data if len(row) > 1]  # Ensure we don't include rows with only one cell
    return parsed_data

def convert_to_csv(parsed_data):
    if not parsed_data:
        return "Failed to parse the data. Please ensure the response is in the expected format."
        
            
    # Check for mismatched column counts
    expected_columns = 4
    mismatched_rows = [index for index, row in enumerate(parsed_data, start=1) if len(row) != expected_columns]

    if mismatched_rows:
        # Print the mismatched rows for debugging
        for index in mismatched_rows:
            return f"Row {index}: {parsed_data[index-1]}"  # index-1 because mismatched_rows starts from 1
        
        return f"Data mismatch. Expected {expected_columns} columns but found different column counts in rows: {', '.join(map(str, mismatched_rows))}. Please review the data."

    # Assuming the first row contains column headers
    df = pd.DataFrame(parsed_data[1:], columns=parsed_data[0])
    print(df)
    csv_string = df.to_csv(index=False)
    return csv_string

def split_into_segments(self, text, max_tokens = 3800):
    sentences = sent_tokenize(text)
    segments = []
    segment = ""
    segment_tokens = 0

    for sentence in sentences:
        num_tokens = len(word_tokenize(sentence))
        if segment_tokens + num_tokens > max_tokens:
            segments.append(segment.strip())
            segment = sentence
            segment_tokens = num_tokens
        else:
            segment += " " + sentence
            segment_tokens += num_tokens

    if segment:
        segments.append(segment.strip())

    return segments
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return 'This is a simple Flask web application.'


@app.route('/api/call_gpt',methods=['POST'])
def api_call_gpt():
    data_content = request.form['data_content']
    num_themes = request.form['num_themes']
    custom_prompt = request.form['custom_prompt']
    data_type = request.form['data_type']
    data_type=data_type_dict[data_type]

    if len(data_content) > 4096:
        dataset_segments = split_into_segments(data_content, 4096 - 1500)  # Reserve some tokens for additional prompts
        # 不要在这里提交分段，只是保存它们
        saved_segments = dataset_segments
        # QMessageBox.information(self, "Success", "Dataset has been segmented and is ready for analysis.")
    else:
        # 如果数据内容不超过4096 tokens
        saved_segments = [data_content]
    print('get form')
    print(request.form)
    # print(saved_segments)
    result = call_chatgpt(num_themes=num_themes,prompt=prompts[data_type],data_content=data_content,
                 saved_segments=saved_segments,custom_prompt=custom_prompt)
    # result=[test_result]
    # print('result')
    # print(result)
    csv=''
    # prased_data = parse_response_to_csv(result[0])
    # csv = convert_to_csv(prased_data)
    # print('csv')
    # print(csv)
    return {'code':200,'message':result,'csv_string':csv}
    

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {'code':400,'message':'No file part'}
    file = request.files['file']
    if file.filename == '':
        return {'code':400,'message':'No selected file'}
    if file:
        # filename = secure_filename(file.filename)
        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(file)
        msg = extract_file(file)
        print(msg)
        return msg
        # return render_template('data.html',data=sdata)

def extract_file(file):
    filename = file.filename

    try:
        if filename.endswith('.csv'):
            data = pd.read_csv(file, encoding='utf-8')
        elif filename.endswith('.xlsx'):
            data = pd.read_excel(file, engine='openpyxl')
        elif filename.endswith('.docx'):
            doc = Document(file)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            data = pd.DataFrame(full_text, columns=['Content'])
        else:
            return {'code':400,'message':'Unsupported file type'}
        
        sdata = '\n'.join(data.apply(lambda row: ' '.join(row.astype(str)), axis=1))
        headers = list(data.columns)
        print(headers)
        return {'code':200,'message':{'data':sdata,'headers':headers}}
    except UnicodeDecodeError:
        print('unicode')
        try:
            if filename.endswith('.csv'):
                data = pd.read_csv(file, encoding='ISO-8859-1')
            elif filename.endswith('.xlsx'):
                data = pd.read_excel(file, engine='openpyxl')
            print('read')
            return {'code':200,'message':{'data':sdata,'headers':headers}}
        except Exception as e:
            print('woos')
            return {'code':400,'message':f"Unrecognized file format or encoding issue: {str(e)}"},
    except Exception as err:
        print(err)
        return {'code':400,'message':f"Unrecognized file format or encoding issue: {str(err)}"},
    
    return {'code':400,'message':'Operation failed'}
if __name__ == '__main__':
    app.run(debug=True,port=80)