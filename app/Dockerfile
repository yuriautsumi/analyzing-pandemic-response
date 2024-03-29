FROM tiangolo/uwsgi-nginx-flask:python3.9

RUN mkdir /code
WORKDIR /code
ADD app/requirements.txt /code/
RUN pip install -r /code/requirements.txt --no-cache-dir
ADD app/ /code/

# ssh
ENV SSH_PASSWD "root:Docker!"
RUN apt-get update \
        && apt-get install -y --no-install-recommends dialog \
        && apt-get update \
	&& apt-get install -y --no-install-recommends openssh-server \
	&& echo "$SSH_PASSWD" | chpasswd 

COPY app/sshd_config /etc/ssh/
COPY app/init.sh /usr/local/bin/
	
RUN chmod u+x /usr/local/bin/init.sh
EXPOSE 8000 2222
#CMD ["python", "/code/manage.py", "runserver", "0.0.0.0:8000"]
ENTRYPOINT ["/usr/local/bin/init.sh"]

