FROM ubuntu:22.04
COPY entrypoint.sh /entrypoint.sh
RUN chmod u+x /entrypoint.sh
CMD ["/entrypoint.sh"]
