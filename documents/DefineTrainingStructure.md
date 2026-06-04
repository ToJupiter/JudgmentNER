1. First of all, I will let you understand what our data is:
``` json
  {
    "file_name": "2ta1941690t1cvn_visual.md",
    "laws_cited": [
      "khoản tiền nợ do thế chấp ngôi nhà số H T cho Ngân hàng (cả gốc và lãi khoảng hơn 5.",
      "khoản 4 Điều 174 Bộ luật Hình sự.",
      "khoản 4 Điều 174, điểm b, s, v khoản 1, khoản 2 Điều 51; của Bộ luật Hình sự, đề nghị xử phạt bị cáo Võ Thị Kiều L từ 12 (mười hai) đến 14 (mười bốn) năm tù.",
      "khoản 4 Điều 174, b, s, v khoản 1, khoản 2 Điều 51; Điều 54 của Bộ luật Hình sự đề nghị xử phạt bị cáo Phạm Văn Đ từ 07 (mười hai) đến 08 (mười bốn) năm tù.",
      "Điều 48 Bộ luật hình sự, các Điều 584, 585, 586, 587, 589 của Bộ luật dân sự, đề nghị buộc các bị cáo Võ Thị Kiều L, Phạm Văn Đ phải bồi thường cho Vợ chồng bà Ban Gia Q, Phạm Xuân H5 7.",
      "các Điều 47, 48 của Bộ luật hình sự, đê106 của Bộ luật tố tụng hình sự, đề nghị: Trả lại 01 điện thoại di động hiệu Iphone cho Phạm Văn Đ, nhưng tạm giữ để đảm bảo cho việc thi hành án.",
      "khoản 1, khoản 2 Điều 51, Điều 54 của Bộ luật hình sự, xử phạt bị cáo với mức hình phạt thấp nhất.",
      "khoản 1, khoản 2 Điều 51, Điều 54 của Bộ luật hình sự, xử phạt bị cáo với mức hình phạt thấp hơn mức hình phạt mà đại diện VKS yêu cầu.",
      "khoản 4 Điều 174 Bộ luật Hình sự là đúng người, đúng tội và có căn cứ pháp luật.",
      "khoản 1, khoản 2 Điều 51 của Bộ luật Hình sự nên các bị cáo được giảm nhẹ một phần hình phạt.",
      "các Điều 135, 136 Bộ luật Tố tụng hình sự và điểm a khoản 1 Điều 23 Nghị quyết 326/2016/UBTVQH14 ngày 30 tháng 12 năm 2016 của Ủy ban thường vụ Quốc hội khóa 14 quy định về án phí lệ phí Tòa án, các bị cáo Võ Thị Kiều L, Phạm Văn Đ, mỗi bị cáo phải chịu 200.",
      "khoản 4 Điều 174 , điểm b, s, v khoản 1, khoản 2 Điều 51, Điều 65 của Bộ luật Hình sự.",
      "khoản 4 Điều 174 , điểm b, s, v khoản 1, khoản 2 Điều 51 Điều 65 của Bộ luật Hình sự.",
      "Điều 48 BLHS.",
      "các Điều 47, 48 Bộ luật hình Sự, Điều 106 Bộ luật Tố tụng hình sự: 4.",
      "Điều 135, 136 Bộ luật Tố tụng hình sự, điểm a, c, đ khoản 1 Điều 23 Nghị quyết số: 326/2016/UBTVQH14 ngày 30 tháng 12 năm 2016 của Ủy ban thường vụ Quốc hội khóa 14.",
      "khoản 2 Điều 468 của BLDS (2015) tương ứng với số tiền và thời gian chưa thi hành án.",
      "Điều 2 Luật Thi hành án dân sự thì người được thi hành án dân sự, người phải thi hành án dân sự có quyền thỏa thuận thi hành án, quyền yêu cầu thi hành án, tự nguyện thi hành án 9 hoặc bị cưỡng chế thi hành án theo quy định tại các điều 6, 7 và 9 Luật Thi hành án dân sự; thời hiệu thi hành án được thực hiện theo quy định tại Điều 30 Luật thi hành án dân sự."
    ],
    "count": 18
  },
```
2. This is the data gathered by regex, but it is extremely wrong. Which means we have to use a customized NER model to run NER on the regex-ed results. We need to extract some informations: ĐIỀU, TÊN LUẬT + NĂM LUẬT. For example: "Điều 135, 136 Bộ luật Tố tụng hình sự". We need to know what is the laws and what is the name of laws. Other cases điểm a khoản 1 Điều 23 Nghị quyết 326/2016/UBTVQH14 ngày 30 tháng 12 năm 2016 của Ủy ban thường vụ Quốc hội khóa 14 quy định về án phí lệ phí Tòa án". We need to seperate out what is the Điều number and what is the law name, exactly. The law name in here replicates multiple times, so we can overfit the model, it is quite ok.
3. Given the data above, define me what a NER model needs to do. We will use CNN, and for now having around 70k data so we will split into random train & test and also do cross validation to test. You need to define the state of the CNN NER model, its params count, etc... The input string for the model is like above length (not too long, max ~500 tokens). But we need to inference fast, on AVX only processor (no AVX2). So we need to optimize things well and the inference speed must be fast. DEFINE THE STATE TO TRAIN AND INFERE CAREFULLY AS THIS WILL AFFECT HOW I USE DEEPSEEK TO LABEL IT LATER.
4. Define the way that we must use to use DeepSeek API to label those data. How to label them? I use DeepSeek v4 Flash, optimize the prompt for RIGHTFULLY LABELLING AND 128 ASYNCIO THREADS TO DEEPSEEK API AT A TIME. MAYBE WE NEED TO PUT IN ONE "laws_cited" AT A TIME FOR THE PROMPT TO RUN QUICKLY. GENERATE ME THE CLEAN CODE FOR DEEPSEEK. IT MUST BE RUNNING RIGHT NOW, NO COMMENTS IN CODE AND 128 ASYNCIO THREADS WITH LOAD_DOTENV. 
``` py
from openai import OpenAI

# for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
client = OpenAI(api_key="<your API key>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
  ],
    max_tokens=1024,
    temperature=0.7,
    stream=False
)

print(response.choices[0].message.content)
```

``` json
{
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "Hi",
      "role": "user"
    }
  ],
  "model": "deepseek-v4-pro",
  "thinking": {
    "type": "enabled"
  },
  "reasoning_effort": "high",
  "max_tokens": 4096,
  "response_format": {
    "type": "text"
  },
  "stop": null,
  "stream": false,
  "stream_options": null,
  "temperature": 1,
  "top_p": 1,
  "tools": null,
  "tool_choice": "none",
  "logprobs": false,
  "top_logprobs": null
}
```

Also hard cases like this:
```
 "Khoản 1 Điều 28, khoản 3 Điều 26, điểm a khoản 1 Điều 39, khoản 4 Điều 147 của Bộ luật tố tụng dân sự năm 2015; Khoản 2 Điều 1 sửa đổi, bổ sung Điều 35 của Luật số 85/2025/QH15 sửa đổi, bổ sung một số điều của Bộ luật tố tụng dân sự, Luật tố tụng hành chính, Luật tư pháp người chưa thành niên, Luật phá sản và Luật hoà giải, đối thoại tại Toà án ngày 25/6/2025; Điều 27, khoản 2 Điều 33, khoản 2 Điều 37, Điều 59, khoản 2 Điều 81, Điều 84 Luật Hôn nhân và Gia đình năm 2014; Điều 466, khoản 2 Điều 468 Bộ luật dân sự năm 2015; Điều 12 và Điều 27 Nghị quyết số 326/2017/UBTVQH14 ngày 30/12/2016 của Ủy ban thường vụ Quốc hội quy định về mức thu, miễn, giảm, thu, nộp, quản lý và sử dụng án phí và lệ phí Tòa án.",
```

Use the knowledge of Vietnamese law system to ensure that these cases are handled properly. Also, multiple laws and multiple articles can be cited in one sentence. JSON costs a bunch of tokens, find a creative way to make DeepSeek label right but costs less and easy to train the NER. Please express everything deeply. Cache the system prompt to DeepSeek to save tokens, modify the user prompt (user prompt is the law input to be classified by DeepSeek). This is the cache documentation of DeepSeek:
```
Each user request will trigger the construction of a hard disk cache. If subsequent requests have overlapping prefixes with previous requests, the overlapping part will only be fetched from the cache, which counts as a "cache hit."

Checking Cache Hit Status

In the response from the DeepSeek API, we have added two fields in the usage section to reflect the cache hit status of the request:

    prompt_cache_hit_tokens: The number of tokens in the input of this request that resulted in a cache hit.

    prompt_cache_miss_tokens: The number of tokens in the input of this request that did not result in a cache hit.

```

You should may check this for us.