{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c2775c4-98e6-4d6c-9f62-8b578e5ee2a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "load_dotenv()\n",
    "\n",
    "TELEGRAM_TOKEN = os.getenv(\"TELEGRAM_BOT_TOKEN\")\n",
    "GEMINI_KEY = os.getenv(\"GOOGLE_API_KEY\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
