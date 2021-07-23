from django.shortcuts import render,redirect
from .models import login
from django.contrib.auth.models import auth
from django.contrib import messages
from django.http import HttpResponse

# Create your views here.
def login(request):
    if request.method == 'POST':
        username = request.POST["username"]

        password = request.POST["pass"]
        user = auth.authenticate(username = username, password = password)
        if user is not None:
            auth.login(request, user)
            messages.info(request, 'Successfull!!')
            return redirect('userhome')
        else:
            messages.info(request, 'Invalid!!')
            return redirect('login')
    return render(request, 'login.html')

