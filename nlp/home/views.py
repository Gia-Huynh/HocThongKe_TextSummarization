from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse as HttpJsonResponse
import json
from .models import Works
import datetime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os
from django.conf import settings


# Create your views here.
def home(request):
    return render(request, "home/home.html")


def get_model(request):
    return render(request, "")


def inteference(input_text):
    # Define the path to your saved model
    checkpoint = os.path.join(
        settings.BASE_DIR, "home", "my_awesome_billsum_model", "checkpoint-500"
    )

    # Load the trained model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)

    # Tokenize the input text
    inputs = tokenizer.encode(
        "summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True
    )

    # Generate summary (you can tweak the parameters like max_length and num_beams as needed)
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )

    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


@csrf_exempt
def get_input(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body.decode("utf-8"))
            input_text = data.get("text")

            if not input_text:
                return HttpJsonResponse({"error": "No input text provided"}, status=400)

            text_summary = inteference(input_text)
            print("Text Summary:", text_summary)  # Log summary

            time_save = datetime.datetime.now()
            data_save = Works.objects.create(
                text=input_text, text_summary=text_summary, data_save=time_save
            )
            data_save.save()
            response_data = {
                "text_summary": text_summary  # Include the summary in the response data
            }

            return HttpJsonResponse(response_data)
        except json.JSONDecodeError:
            return HttpJsonResponse({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            print("Error:", str(e))  # Log any exception
            return HttpJsonResponse({"error": str(e)}, status=500)
    else:
        return HttpJsonResponse({"error": "Invalid request method"}, status=405)


@csrf_exempt
def get_history(request):
    if request.method == "POST" or request.method == "GET":
        data = Works.objects.all()
        data = list(data.values())
        data = data[::-1]

        return render(request, "history/history.html", {"data": data})
