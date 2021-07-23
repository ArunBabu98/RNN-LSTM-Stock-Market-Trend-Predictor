from django.shortcuts import render, redirect
from django.contrib.auth.forms import User

# Create your views here.

def register(request):
    if request.method == 'POST':
        user = request.POST['username']
        pwd = request.POST['pass']
        user = User.objects.create_user(username=user,password=pwd)
        user.save()
        return redirect('login')
    else:
        return render(request,'register.html')